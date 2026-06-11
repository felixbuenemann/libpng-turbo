/* write_filter_neon_intrinsics.c - NEON optimised write filter trials
 *
 * Copyright (c) 2026 Cosmin Truta
 * Derived from intel/write_filter_sse2_intrinsics.c
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * This file is included by arm_init.c after filter_neon_intrinsics.c and
 * uses the paeth() helper defined there.  See the comments at the top of
 * intel/write_filter_sse2_intrinsics.c for the design; in short, the write
 * filter trials are pure maps (no inter-pixel dependence), so a single
 * implementation processing 16 bytes per iteration covers every pixel
 * size, with scalar tails because the write buffers have no padding.
 */

/* The score of a filtered byte is (v < 128 ? v : 256 - v), which is the
 * absolute value of the byte read as a signed two's complement number; the
 * vector form is min(v, -v) accumulated with pairwise-add-long steps.
 */
#define png_setup_score(v) ((v) < 128 ? (v) : 256 - (v))

static uint32x4_t
png_wfilter_score_neon(uint32x4_t acc, uint8x16_t d)
{
   uint8x16_t s = vminq_u8(d, vsubq_u8(vdupq_n_u8(0), d));
   return vpadalq_u16(acc, vpaddlq_u8(s));
}

static size_t
png_wfilter_fold_neon(uint32x4_t acc)
{
   uint64x2_t t = vpaddlq_u32(acc);
   return (size_t)(vgetq_lane_u64(t, 0) + vgetq_lane_u64(t, 1));
}

/* The abandonment check (sum > lmins) runs once per block of this many
 * bytes; any sum exceeding lmins is rejected by the caller no matter by
 * how much, so the block size does not affect the selected filter.  The
 * uint32x4_t score accumulator cannot overflow within a block (each lane
 * gains at most 512 per 16 bytes).
 */
#define PNG_WFILTER_BLOCK 4096U

static size_t
png_setup_sub_row_neon(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   png_byte *dp = png_ptr->try_row + 1;
   size_t i;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_sub_row_neon");

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
      uint32x4_t acc = vdupq_n_u32(0);

      for (; i + 16 <= stop; i += 16)
      {
         uint8x16_t x = vld1q_u8(rp + i);
         uint8x16_t a = vld1q_u8(rp + i - bpp);
         uint8x16_t d = vsubq_u8(x, a);

         vst1q_u8(dp + i, d);
         acc = png_wfilter_score_neon(acc, d);
      }

      sum += png_wfilter_fold_neon(acc);

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
png_setup_up_row_neon(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   const png_byte *pp = png_ptr->prev_row + 1;
   png_byte *dp = png_ptr->try_row + 1;
   size_t i = 0;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_up_row_neon");

   PNG_UNUSED(bpp) /* the up filter does not use the pixel to the left */

   png_ptr->try_row[0] = PNG_FILTER_VALUE_UP;

   while (i + 16 <= row_bytes)
   {
      size_t stop = row_bytes - i > PNG_WFILTER_BLOCK ?
         i + PNG_WFILTER_BLOCK : row_bytes;
      uint32x4_t acc = vdupq_n_u32(0);

      for (; i + 16 <= stop; i += 16)
      {
         uint8x16_t x = vld1q_u8(rp + i);
         uint8x16_t b = vld1q_u8(pp + i);
         uint8x16_t d = vsubq_u8(x, b);

         vst1q_u8(dp + i, d);
         acc = png_wfilter_score_neon(acc, d);
      }

      sum += png_wfilter_fold_neon(acc);

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
png_setup_avg_row_neon(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   const png_byte *pp = png_ptr->prev_row + 1;
   png_byte *dp = png_ptr->try_row + 1;
   size_t i;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_avg_row_neon");

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
      uint32x4_t acc = vdupq_n_u32(0);

      for (; i + 16 <= stop; i += 16)
      {
         uint8x16_t x = vld1q_u8(rp + i);
         uint8x16_t a = vld1q_u8(rp + i - bpp);
         uint8x16_t b = vld1q_u8(pp + i);

         /* vhaddq_u8 is the truncating average PNG requires. */
         uint8x16_t d = vsubq_u8(x, vhaddq_u8(a, b));

         vst1q_u8(dp + i, d);
         acc = png_wfilter_score_neon(acc, d);
      }

      sum += png_wfilter_fold_neon(acc);

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
png_setup_paeth_row_neon(png_struct *png_ptr, png_uint_32 bpp,
    size_t row_bytes, size_t lmins)
{
   const png_byte *rp = png_ptr->row_buf + 1;
   const png_byte *pp = png_ptr->prev_row + 1;
   png_byte *dp = png_ptr->try_row + 1;
   size_t i;
   size_t sum = 0;
   unsigned int v;

   png_debug(1, "in png_setup_paeth_row_neon");

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
      uint32x4_t acc = vdupq_n_u32(0);

      for (; i + 16 <= stop; i += 16)
      {
         uint8x16_t a = vld1q_u8(rp + i - bpp);
         uint8x16_t b = vld1q_u8(pp + i);
         uint8x16_t c = vld1q_u8(pp + i - bpp);
         uint8x16_t x = vld1q_u8(rp + i);

         /* There is no carried dependence, so the Paeth predictor runs on
          * 16 independent bytes per iteration using the paeth() helper
          * from filter_neon_intrinsics.c on each 8 byte half.
          */
         uint8x16_t pred = vcombine_u8(
            paeth(vget_low_u8(a), vget_low_u8(b), vget_low_u8(c)),
            paeth(vget_high_u8(a), vget_high_u8(b), vget_high_u8(c)));
         uint8x16_t d = vsubq_u8(x, pred);

         vst1q_u8(dp + i, d);
         acc = png_wfilter_score_neon(acc, d);
      }

      sum += png_wfilter_fold_neon(acc);

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
