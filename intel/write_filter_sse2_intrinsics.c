/* write_filter_sse2_intrinsics.c - SSE2 optimised write filter trials
 *
 * Copyright (c) 2026 Cosmin Truta
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * This file is included by intel_init.c after filter_sse2_intrinsics.c and
 * uses the helpers defined there (load8, abs_i16, if_then_else).
 *
 * The write filter trials compute a candidate filtered row into
 * png_struct::try_row and return its "minimum sum of absolute differences"
 * score.  Unlike the read filters there is no inter-pixel dependence (the
 * input is the unfiltered row), so a single implementation processing 16
 * bytes per iteration covers every pixel size; 'bpp' is only a load offset.
 *
 * The write buffers have no padding, so the vector loops stop at the last
 * whole 16 byte block and the remainder is handled by the scalar tails.
 */

/* The score of a filtered byte is (v < 128 ? v : 256 - v), which is the
 * absolute value of the byte read as a signed two's complement number; the
 * vector form is min(v, -v) summed with the SAD instruction.
 */
#define png_setup_score(v) ((v) < 128 ? (v) : 256 - (v))

static __m128i
png_wfilter_score_sse2(__m128i acc, __m128i d)
{
   const __m128i zero = _mm_setzero_si128();
   __m128i s = _mm_min_epu8(d, _mm_sub_epi8(zero, d));
   return _mm_add_epi64(acc, _mm_sad_epu8(s, zero));
}

static size_t
png_wfilter_fold_sse2(__m128i acc)
{
   __m128i t = _mm_add_epi64(acc, _mm_unpackhi_epi64(acc, acc));
   png_uint_32 lo = (png_uint_32)_mm_cvtsi128_si32(t);
   png_uint_32 hi = (png_uint_32)_mm_cvtsi128_si32(_mm_srli_si128(t, 4));

   /* The sum always fits in size_t (png_write_find_filter only scores rows
    * shorter than PNG_SIZE_MAX/128), so 'hi' is zero when size_t is 32 bits;
    * the double shift avoids an illegal 32-bit shift count.
    */
   return (size_t)lo + (((size_t)hi << 16) << 16);
}

/* The abandonment check (sum > lmins) runs once per block of this many
 * bytes; any sum exceeding lmins is rejected by the caller no matter by
 * how much, so the block size does not affect the selected filter.
 */
#define PNG_WFILTER_BLOCK 4096U

static size_t
png_setup_sub_row_sse2(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   png_byte *dp = png_ptr->try_row + 1;
   size_t i;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_sub_row_sse2");

   png_ptr->try_row[0] = PNG_FILTER_VALUE_SUB;

   for (i = 0; i < bpp; i++)
   {
      v = dp[i] = rp[i];
      sum += png_setup_score(v);
   }

   while (i + 16 <= row_bytes)
   {
      size_t stop = row_bytes - i > PNG_WFILTER_BLOCK ?
         i + PNG_WFILTER_BLOCK : row_bytes;
      __m128i acc = _mm_setzero_si128();

      for (; i + 16 <= stop; i += 16)
      {
         __m128i x = _mm_loadu_si128((const __m128i *)(rp + i));
         __m128i a = _mm_loadu_si128((const __m128i *)(rp + i - bpp));
         __m128i d = _mm_sub_epi8(x, a);

         _mm_storeu_si128((__m128i *)(dp + i), d);
         acc = png_wfilter_score_sse2(acc, d);
      }

      sum += png_wfilter_fold_sse2(acc);

      if (sum > lmins) /* We are already worse, don't continue. */
         return sum;
   }

   for (; i < row_bytes; i++)
   {
      v = dp[i] = 0xffU & (unsigned int)(rp[i] - rp[i - bpp]);
      sum += png_setup_score(v);
   }

   return sum;
}

static size_t
png_setup_up_row_sse2(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   const png_byte *pp = png_ptr->prev_row + 1;
   png_byte *dp = png_ptr->try_row + 1;
   size_t i = 0;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_up_row_sse2");

   PNG_UNUSED(bpp) /* the up filter does not use the pixel to the left */

   png_ptr->try_row[0] = PNG_FILTER_VALUE_UP;

   while (i + 16 <= row_bytes)
   {
      size_t stop = row_bytes - i > PNG_WFILTER_BLOCK ?
         i + PNG_WFILTER_BLOCK : row_bytes;
      __m128i acc = _mm_setzero_si128();

      for (; i + 16 <= stop; i += 16)
      {
         __m128i x = _mm_loadu_si128((const __m128i *)(rp + i));
         __m128i b = _mm_loadu_si128((const __m128i *)(pp + i));
         __m128i d = _mm_sub_epi8(x, b);

         _mm_storeu_si128((__m128i *)(dp + i), d);
         acc = png_wfilter_score_sse2(acc, d);
      }

      sum += png_wfilter_fold_sse2(acc);

      if (sum > lmins) /* We are already worse, don't continue. */
         return sum;
   }

   for (; i < row_bytes; i++)
   {
      v = dp[i] = 0xffU & (unsigned int)(rp[i] - pp[i]);
      sum += png_setup_score(v);
   }

   return sum;
}

static size_t
png_setup_avg_row_sse2(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   const png_byte *pp = png_ptr->prev_row + 1;
   png_byte *dp = png_ptr->try_row + 1;
   const __m128i ones = _mm_set1_epi8(1);
   size_t i;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_avg_row_sse2");

   png_ptr->try_row[0] = PNG_FILTER_VALUE_AVG;

   for (i = 0; i < bpp; i++)
   {
      v = dp[i] = 0xffU & (unsigned int)(rp[i] - pp[i] / 2);
      sum += png_setup_score(v);
   }

   while (i + 16 <= row_bytes)
   {
      size_t stop = row_bytes - i > PNG_WFILTER_BLOCK ?
         i + PNG_WFILTER_BLOCK : row_bytes;
      __m128i acc = _mm_setzero_si128();

      for (; i + 16 <= stop; i += 16)
      {
         __m128i x = _mm_loadu_si128((const __m128i *)(rp + i));
         __m128i a = _mm_loadu_si128((const __m128i *)(rp + i - bpp));
         __m128i b = _mm_loadu_si128((const __m128i *)(pp + i));

         /* PNG requires a truncating average; pavgb rounds up, so subtract
          * the rounding fixup exactly as in the read filters above.
          */
         __m128i avg = _mm_avg_epu8(a, b);
         __m128i d;
         avg = _mm_sub_epi8(avg, _mm_and_si128(_mm_xor_si128(a, b), ones));
         d = _mm_sub_epi8(x, avg);

         _mm_storeu_si128((__m128i *)(dp + i), d);
         acc = png_wfilter_score_sse2(acc, d);
      }

      sum += png_wfilter_fold_sse2(acc);

      if (sum > lmins) /* We are already worse, don't continue. */
         return sum;
   }

   for (; i < row_bytes; i++)
   {
      v = dp[i] = 0xffU & (unsigned int)(rp[i] - (pp[i] + rp[i - bpp]) / 2);
      sum += png_setup_score(v);
   }

   return sum;
}

static size_t
png_setup_paeth_row_sse2(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   const png_byte *pp = png_ptr->prev_row + 1;
   png_byte *dp = png_ptr->try_row + 1;
   const __m128i zero = _mm_setzero_si128();
   size_t i;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_paeth_row_sse2");

   png_ptr->try_row[0] = PNG_FILTER_VALUE_PAETH;

   for (i = 0; i < bpp; i++)
   {
      v = dp[i] = 0xffU & (unsigned int)(rp[i] - pp[i]);
      sum += png_setup_score(v);
   }

   while (i + 16 <= row_bytes)
   {
      size_t stop = row_bytes - i > PNG_WFILTER_BLOCK ?
         i + PNG_WFILTER_BLOCK : row_bytes;
      __m128i acc = _mm_setzero_si128();

      for (; i + 16 <= stop; i += 16)
      {
         __m128i a8 = _mm_loadu_si128((const __m128i *)(rp + i - bpp));
         __m128i b8 = _mm_loadu_si128((const __m128i *)(pp + i));
         __m128i c8 = _mm_loadu_si128((const __m128i *)(pp + i - bpp));
         __m128i x = _mm_loadu_si128((const __m128i *)(rp + i));
         __m128i pred_lo = zero, pred_hi = zero, d;
         int half;

         /* There is no carried dependence, so unlike the read filters the
          * Paeth predictor runs on 16 independent bytes per iteration, as
          * two 8 byte halves with 16-bit intermediates.
          */
         for (half = 0; half < 2; half++)
         {
            __m128i a, b, c, pa, pb, pc, smallest, nearest;

            if (half == 0)
            {
               a = _mm_unpacklo_epi8(a8, zero);
               b = _mm_unpacklo_epi8(b8, zero);
               c = _mm_unpacklo_epi8(c8, zero);
            }
            else
            {
               a = _mm_unpackhi_epi8(a8, zero);
               b = _mm_unpackhi_epi8(b8, zero);
               c = _mm_unpackhi_epi8(c8, zero);
            }

            /* (p-a) == (a+b-c - a) == (b-c) */
            pa = _mm_sub_epi16(b, c);

            /* (p-b) == (a+b-c - b) == (a-c) */
            pb = _mm_sub_epi16(a, c);

            /* (p-c) == (a+b-c - c) == (b-c)+(a-c) */
            pc = _mm_add_epi16(pa, pb);

            pa = abs_i16(pa);  /* |p-a| */
            pb = abs_i16(pb);  /* |p-b| */
            pc = abs_i16(pc);  /* |p-c| */

            smallest = _mm_min_epi16(pc, _mm_min_epi16(pa, pb));

            /* Paeth breaks ties favoring a over b over c. */
            nearest = if_then_else(_mm_cmpeq_epi16(smallest, pa), a,
                      if_then_else(_mm_cmpeq_epi16(smallest, pb), b,
                                                                  c));

            if (half == 0)
               pred_lo = nearest;
            else
               pred_hi = nearest;
         }

         d = _mm_sub_epi8(x, _mm_packus_epi16(pred_lo, pred_hi));

         _mm_storeu_si128((__m128i *)(dp + i), d);
         acc = png_wfilter_score_sse2(acc, d);
      }

      sum += png_wfilter_fold_sse2(acc);

      if (sum > lmins) /* We are already worse, don't continue. */
         return sum;
   }

   for (; i < row_bytes; i++)
   {
      int a, b, c, pa, pb, pc, p;

      b = pp[i];
      c = pp[i - bpp];
      a = rp[i - bpp];

      p = b - c;
      pc = a - c;

#ifdef PNG_USE_ABS
      pa = abs(p);
      pb = abs(pc);
      pc = abs(p + pc);
#else
      pa = p < 0 ? -p : p;
      pb = pc < 0 ? -pc : pc;
      pc = (p + pc) < 0 ? -(p + pc) : p + pc;
#endif

      p = (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c;

      v = dp[i] = 0xffU & (unsigned int)(rp[i] - p);
      sum += png_setup_score(v);
   }

   return sum;
}
