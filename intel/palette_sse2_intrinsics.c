/* palette_sse2_intrinsics.c - SSE2 optimised palette expansion functions
 *
 * Copyright (c) 2026 Cosmin Truta
 * Derived from arm/palette_neon_intrinsics.c
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

/* Build an RGBA8 palette from the separate RGB and alpha palettes.
 *
 * Unlike the NEON implementation this is not vectorized: the riffle runs
 * once per image (not per row) over at most 256 entries, and a scalar loop
 * avoids reading beyond the end of the 768 byte RGB palette.
 */
static void
png_riffle_palette_sse2(png_byte *riffled_palette, const png_color *palette,
    const png_byte *trans_alpha, int num_trans)
{
   int i;

   png_debug(1, "in png_riffle_palette_sse2");

   for (i = 0; i < 256; i++)
   {
      riffled_palette[i * 4 + 0] = palette[i].red;
      riffled_palette[i * 4 + 1] = palette[i].green;
      riffled_palette[i * 4 + 2] = palette[i].blue;
      riffled_palette[i * 4 + 3] = i < num_trans ? trans_alpha[i] : 0xff;
   }
}

/* Expand a 1, 2 or 4 bit packed indexed row in place to 8 bit indices,
 * mirroring the bit_depth < 8 phase of png_do_expand_palette.
 *
 * The expansion works from the end of the row backwards so that no packed
 * byte is overwritten before it has been read.  The trailing pixels (a
 * partial vector chunk, including any partial trailing byte) are expanded
 * with the scalar loop; whole chunks of 8 packed bytes are expanded with
 * cascaded shift/mask/interleave stages, each stage splitting every byte
 * into its high and low halves (the leftmost pixel is in the high bits).
 */
static void
png_target_expand_bits_sse2(png_byte *row, png_uint_32 row_width,
    unsigned int bit_depth)
{
   /* Pixels per 8-byte chunk: 16, 32 or 64. */
   const png_uint_32 pixels_per_chunk = 64U / bit_depth;
   const unsigned int mask = (1U << bit_depth) - 1U;
   png_uint_32 simd_px = row_width - (row_width % pixels_per_chunk);
   png_uint_32 i;

   png_debug(1, "in png_target_expand_bits_sse2");

   /* Scalar tail: pixels [simd_px, row_width), written backwards. */
   for (i = row_width; i > simd_px; )
   {
      unsigned int sbyte, shift;
      --i;
      sbyte = row[(i * bit_depth) >> 3];
      shift = (8U - bit_depth) - ((i * bit_depth) & 7U);
      row[i] = (png_byte)((sbyte >> shift) & mask);
   }

   /* Whole chunks, highest first. */
   for (i = simd_px; i > 0; )
   {
      const png_byte *sp;
      png_byte *dp;

      i -= pixels_per_chunk;
      sp = row + ((i * bit_depth) >> 3);
      dp = row + i;

      if (bit_depth == 4)
      {
         __m128i x = load8(sp);
         __m128i nib = _mm_set1_epi8(0x0f);
         __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), nib);
         __m128i lo = _mm_and_si128(x, nib);
         _mm_storeu_si128((__m128i *)dp, _mm_unpacklo_epi8(hi, lo));
      }

      else if (bit_depth == 2)
      {
         __m128i x = load8(sp);
         __m128i nib = _mm_set1_epi8(0x0f);
         __m128i two = _mm_set1_epi8(0x03);
         __m128i q = _mm_unpacklo_epi8(
             _mm_and_si128(_mm_srli_epi16(x, 4), nib),
             _mm_and_si128(x, nib));
         __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 2), two);
         __m128i lo = _mm_and_si128(q, two);
         _mm_storeu_si128((__m128i *)dp, _mm_unpacklo_epi8(hi, lo));
         _mm_storeu_si128((__m128i *)(dp + 16), _mm_unpackhi_epi8(hi, lo));
      }

      else /* bit_depth == 1 */
      {
         __m128i x = load8(sp);
         __m128i nib = _mm_set1_epi8(0x0f);
         __m128i two = _mm_set1_epi8(0x03);
         __m128i one = _mm_set1_epi8(0x01);
         __m128i q = _mm_unpacklo_epi8(
             _mm_and_si128(_mm_srli_epi16(x, 4), nib),
             _mm_and_si128(x, nib));
         __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 2), two);
         __m128i lo = _mm_and_si128(q, two);
         __m128i p0 = _mm_unpacklo_epi8(hi, lo);
         __m128i p1 = _mm_unpackhi_epi8(hi, lo);
         hi = _mm_and_si128(_mm_srli_epi16(p0, 1), one);
         lo = _mm_and_si128(p0, one);
         _mm_storeu_si128((__m128i *)dp, _mm_unpacklo_epi8(hi, lo));
         _mm_storeu_si128((__m128i *)(dp + 16), _mm_unpackhi_epi8(hi, lo));
         hi = _mm_and_si128(_mm_srli_epi16(p1, 1), one);
         lo = _mm_and_si128(p1, one);
         _mm_storeu_si128((__m128i *)(dp + 32), _mm_unpacklo_epi8(hi, lo));
         _mm_storeu_si128((__m128i *)(dp + 48), _mm_unpackhi_epi8(hi, lo));
      }
   }
}

/* Expands a palettized row into RGBA8. */
static png_uint_32
png_target_do_expand_palette_rgba8_sse2(const png_uint_32 *riffled_palette,
    png_uint_32 row_width, const png_byte **ssp, png_byte **ddp)
{
   const png_uint_32 pixels_per_chunk = 4;
   png_uint_32 i;

   png_debug(1, "in png_do_expand_palette_rgba8_sse2");

   if (row_width < pixels_per_chunk)
      return 0;

   /* This function originally gets the last byte of the output row.
    * The SSE2 part writes forward from a given position, so we have
    * to seek this back by 4 pixels x 4 bytes.
    */
   *ddp = *ddp - (pixels_per_chunk * 4 - 1);

   for (i = 0; i + pixels_per_chunk <= row_width; i += pixels_per_chunk)
   {
      /* Gather the four RGBA8 palette entries into one register; SSE has
       * no small-element gather, however on every implementation to date
       * separate 32-bit loads combined with unpack instructions perform
       * well (the loads issue in parallel).
       */
      const png_byte *sp = *ssp - i;
      png_byte *dp = *ddp - i * 4;
      __m128i lo = _mm_unpacklo_epi32(
          _mm_cvtsi32_si128((int)riffled_palette[*(sp - 3)]),
          _mm_cvtsi32_si128((int)riffled_palette[*(sp - 2)]));
      __m128i hi = _mm_unpacklo_epi32(
          _mm_cvtsi32_si128((int)riffled_palette[*(sp - 1)]),
          _mm_cvtsi32_si128((int)riffled_palette[*(sp - 0)]));
      _mm_storeu_si128((__m128i *)dp, _mm_unpacklo_epi64(lo, hi));
   }

   /* Undo the pre-adjustment of *ddp before the pointer handoff,
    * so the scalar fallback in pngrtran.c receives a dp that points
    * to the correct position.
    */
   *ddp = *ddp + (pixels_per_chunk * 4 - 1);
   *ssp = *ssp - i;
   *ddp = *ddp - i * 4;
   return i;
}

/* Expands a palettized row into RGB8.
 *
 * The output pixels are 3 bytes, which SSE cannot store without either
 * over-writing adjacent data or re-shuffling; instead the gathered 32-bit
 * RGBx entries are repacked into three 32-bit words with scalar shifts
 * and stored exactly.  This relies on the target being little-endian,
 * which is always true for this (Intel) target.
 */
static png_uint_32
png_target_do_expand_palette_rgb8_sse2(const png_uint_32 *riffled_palette,
    png_uint_32 row_width, const png_byte **ssp, png_byte **ddp)
{
   const png_uint_32 pixels_per_chunk = 4;
   png_uint_32 i;

   png_debug(1, "in png_do_expand_palette_rgb8_sse2");

   if (row_width < pixels_per_chunk)
      return 0;

   /* Seeking this back by 4 pixels x 3 bytes. */
   *ddp = *ddp - (pixels_per_chunk * 3 - 1);

   for (i = 0; i + pixels_per_chunk <= row_width; i += pixels_per_chunk)
   {
      const png_byte *sp = *ssp - i;
      png_byte *dp = *ddp - i * 3;
      png_uint_32 e0 = riffled_palette[*(sp - 3)];
      png_uint_32 e1 = riffled_palette[*(sp - 2)];
      png_uint_32 e2 = riffled_palette[*(sp - 1)];
      png_uint_32 e3 = riffled_palette[*(sp - 0)];

      /* Each entry has the byte layout [R,G,B,x]; produce the 12 output
       * bytes R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 as three little-endian
       * 32-bit words.
       */
      png_uint_32 w0 = (e0 & 0xffffffU) | (e1 << 24);
      png_uint_32 w1 = ((e1 >> 8) & 0xffffU) | (e2 << 16);
      png_uint_32 w2 = ((e2 >> 16) & 0xffU) | (e3 << 8);

      memcpy(dp, &w0, 4);
      memcpy(dp + 4, &w1, 4);
      memcpy(dp + 8, &w2, 4);
   }

   /* Undo the pre-adjustment of *ddp before the pointer handoff,
    * so the scalar fallback in pngrtran.c receives a dp that points
    * to the correct position.
    */
   *ddp = *ddp + (pixels_per_chunk * 3 - 1);
   *ssp = *ssp - i;
   *ddp = *ddp - i * 3;
   return i;
}
