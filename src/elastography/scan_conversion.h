/***************************************************************************
 Copyright (c) 2012
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.

 ***************************************************************************/

#ifndef _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC
#endif

/* 
 * File:   scan_conversion.h
 * Author: nishikant
 *
 * Created on September 24, 2011, 3:03 AM
 */

#ifndef SCAN_CONVERSION_H
#define	SCAN_CONVERSION_H
#include "common.h"
#include <cuda.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>

#ifdef	__cplusplus
extern "C" {
#endif

    typedef unsigned char data_type;

    int cuda_linear_scan_conversion(data_type *input, data_type **output_t,
            int x, int y, int z, int *display_x, int *display_y, int *display_z,
            float pitch, float speed_of_sound, float sampling_frequency,
            int number_scan_lines);
    data_type * cuda_curved_scan_convert_2(data_type *buffer, int *out_x,
            int *out_y, int *out_z, int x, int y, int z, float radios,
            float ang, float spacing_y, float *final_x);

    data_type * cuda_curved_scan_convert(data_type *buffer, int *out_x,
            int *out_y, int *out_z, int x, int y, int z, float radios,
            float ang);

    int set_cuda_device(int device_id);

    void cuda_rotate_image(data_type *output, data_type *input, int x, int y,
            int z);

    data_type * cuda_average_filter(data_type *buffer, int x, int y, int z,
            int window);

    void cuda_scale_image_mm(data_type_out **out_Im1, int *out_width,
            int *out_height, data_type_out *Im1, int width, int height,
            int no_of_frames, double spacing_x, double spacing_y, float &time1);

    void cuda_malloc_host(void **ptr, size_t ptr_size);

#ifdef	__cplusplus
    }
#endif

#endif	/* SCAN_CONVERSION_H */

