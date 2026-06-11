/* intel_init.c - SSE2 optimized filter functions
 *
 * Copyright (c) 2018 Cosmin Truta
 * Copyright (c) 2016-2017 Glenn Randers-Pehrson
 * Written by Mike Klein and Matt Sarett, Google, Inc.
 * Derived from arm/arm_init.c
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */
#define png_target_impl "intel-sse"

#include "filter_sse2_intrinsics.c"

static void
png_init_filter_functions_sse2(png_struct *pp, unsigned int bpp)
{
   /* The techniques used to implement each of these filters in SSE operate on
    * one pixel at a time.
    * So they generally speed up 3bpp images about 3x, 4bpp images about 4x.
    * They can scale up to 6 and 8 bpp images and down to 2 bpp images,
    * but they'd not likely have any benefit for 1bpp images.
    * Most of these can be implemented using only MMX and 64-bit registers,
    * but they end up a bit slower than using the equally-ubiquitous SSE2.
   */
   png_debug(1, "in png_init_filter_functions_sse2");
   if (bpp == 1)
   {
      /* Only the prefix-sum sub filter applies here (paeth has a
       * specialized 1-byte C implementation).
       */
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub1_sse2;
   }

   else if (bpp == 2)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub2_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg2_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth2_sse2;
   }

   else if (bpp == 3)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub3_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg3_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
         png_read_filter_row_paeth3_sse2;
   }
   else if (bpp == 4)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub4_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg4_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth4_sse2;
   }

   else if (bpp == 6)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub6_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg6_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth6_sse2;
   }

   else if (bpp == 8)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub8_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg8_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth8_sse2;
   }

   /* No need optimize PNG_FILTER_VALUE_UP.  The compiler should
    * autovectorize.
    */
}

#define png_target_init_filter_functions_impl png_init_filter_functions_sse2

#ifdef PNG_TARGET_IMPLEMENTS_WRITE_FILTERS
#include "write_filter_sse2_intrinsics.c"

static void
png_init_write_filter_functions_sse2(png_struct *pp)
{
   png_debug(1, "in png_init_write_filter_functions_sse2");

   /* The write filter trials have no inter-pixel dependences, so these
    * handle every pixel size; 'bpp' is only a load offset.
    */
   pp->write_filter[PNG_FILTER_VALUE_SUB-1] = png_setup_sub_row_sse2;
   pp->write_filter[PNG_FILTER_VALUE_UP-1] = png_setup_up_row_sse2;
   pp->write_filter[PNG_FILTER_VALUE_AVG-1] = png_setup_avg_row_sse2;
   pp->write_filter[PNG_FILTER_VALUE_PAETH-1] = png_setup_paeth_row_sse2;
}

#define png_target_init_write_filter_functions_impl \
   png_init_write_filter_functions_sse2
#endif /* TARGET_IMPLEMENTS_WRITE_FILTERS */

#ifdef PNG_TARGET_STORES_DATA
/*    png_target_free_data_impl
 *       Must be defined if the implementation stores data in
 *       png_struct::target_data.  Need not be defined otherwise.
 */
static void
png_target_free_data_sse2(png_struct *pp)
{
   void *ptr = pp->target_data;
   pp->target_data = NULL;
   png_free(pp, ptr);
}
#define png_target_free_data_impl png_target_free_data_sse2
#endif /* TARGET_STORES_DATA */

#ifdef PNG_TARGET_IMPLEMENTS_EXPAND_PALETTE
/*    png_target_do_expand_palette_impl   [flag: png_target_expand_palette]
 *       static function
 *       OPTIONAL
 *       Handles the transform.  Need not be defined, only called if the
 *       state contains png_target_<transform>, may set this flag to zero, may
 *       return false to indicate that the transform was not done (so the
 *       C implementation must then execute).
 */
#include "palette_sse2_intrinsics.c"

static int
png_target_do_expand_palette_sse2(png_struct *png_ptr, png_row_info *row_info,
    png_byte *row, const png_color *palette, const png_byte *trans_alpha,
    int num_trans)
{
   /* NOTE: it is important that this is done. row_info->width is not a CSE
    * because the pointer is not declared with the 'restrict' parameter, this
    * makes it a CSE but then it is very important that no one changes it in
    * this function, hence the const.
    */
   const png_uint_32 row_width = row_info->width;

   /* This follows the structure of png_target_do_expand_palette_neon in
    * arm/arm_init.c; see the comments there.  The one difference: both the
    * RGBA and the RGB expansions use the "riffled" palette, because reading
    * RGBA8 entries with aligned 32-bit loads is the fastest way to gather
    * palette entries with SSE (and it cannot over-read the 768 byte RGB
    * palette on the last entry).
    */
   if (row_info->color_type == PNG_COLOR_TYPE_PALETTE &&
       row_info->bit_depth <= 8)
   {
      const png_byte *sp = row + (row_width - 1); /* 8 bit palette index */

      if (row_info->bit_depth < 8)
      {
         /* Expand the packed 1, 2 or 4 bit indices to 8 bits first; this
          * also updates row_info so that, should one of the expansions
          * below leave the work to the C implementation (narrow interlaced
          * rows), png_do_expand_palette skips its own unpack phase.
          */
         png_target_expand_bits_sse2(row, row_width, row_info->bit_depth);
         row_info->bit_depth = 8;
         row_info->pixel_depth = 8;
         row_info->rowbytes = row_width;
      }

      /* The riffled palette is initialized here, on demand. */
      if (png_ptr->target_data == NULL)
      {
         /* The data is allocated using png_malloc_warn so the code
          * does not error out on OOM.
          */
         png_ptr->target_data = png_malloc_warn(png_ptr, 256 * 4);

         /* On allocation error it is essential to clear the flag or a
          * massive number of warnings will be output.
          */
         if (png_ptr->target_data != NULL)
            png_riffle_palette_sse2(png_ptr->target_data, palette,
                  trans_alpha, num_trans);
         else
            goto clear_flag;
      }

      if (num_trans > 0)
      {
         /* This is the general convention in the core transform code; when
          * expanding the number of bytes in the row copy down (necessary) and
          * pass a pointer to the last byte, not the first.
          */
         png_byte *dp = row + (4/*RGBA*/*row_width - 1);

         png_uint_32 i = png_target_do_expand_palette_rgba8_sse2(
               png_ptr->target_data, row_width, &sp, &dp);

         if (i == 0) /* nothing was done */
            return 0; /* Return here: interlaced images start out narrow */

         /* Now 'i' may not have reached row_width.
          * NOTE: [i] is not the index into the row buffer, rather it is
          * [row_width-i], this is the way it is done in the original
          * png_do_expand_palette.
          */
         for (; i < row_width; i++)
         {
            if ((int)(*sp) >= num_trans)
               *dp-- = 0xff;
            else
               *dp-- = trans_alpha[*sp];
            *dp-- = palette[*sp].blue;
            *dp-- = palette[*sp].green;
            *dp-- = palette[*sp].red;
            sp--;
         }

         /* Finally update row_info to reflect the expanded output: */
         row_info->bit_depth = 8;
         row_info->pixel_depth = 32;
         row_info->rowbytes = (size_t)row_width * 4;
         row_info->color_type = 6;
         row_info->channels = 4;
         return 1;
      }
      else
      {
         /* No tRNS chunk (num_trans == 0), expand to RGB not RGBA. */
         png_byte *dp = row + (3/*RGB*/ * (size_t)row_width - 1);

         png_uint_32 i = png_target_do_expand_palette_rgb8_sse2(
               png_ptr->target_data, row_width, &sp, &dp);

         if (i == 0)
            return 0; /* Return here: interlaced images start out narrow */

         /* Finish the last bytes: */
         for (; i < row_width; i++)
         {
            *dp-- = palette[*sp].blue;
            *dp-- = palette[*sp].green;
            *dp-- = palette[*sp].red;
            sp--;
         }

         row_info->bit_depth = 8;
         row_info->pixel_depth = 24;
         row_info->rowbytes = (size_t)row_width * 3;
         row_info->color_type = 2;
         row_info->channels = 3;
         return 1;
      }
   }

clear_flag:
   /* Here on malloc failure and on an inapplicable image. */
   png_ptr->target_state &= ~png_target_expand_palette;
   return 0;
}

#define png_target_do_expand_palette_impl png_target_do_expand_palette_sse2
#endif /* TARGET_IMPLEMENTS_EXPAND_PALETTE */
