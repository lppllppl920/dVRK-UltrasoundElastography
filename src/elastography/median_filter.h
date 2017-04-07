/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.
 ***************************************************************************/

#include <stdio.h>
__device__ void fswap(float *a, float *b);
__device__ void fsort_elements(float elements[], int count);

__global__ void moving_average(float *disp, float *target, int moving,
        int width);
__global__ void median_filter(float *disp, float *target, int w, int h,
        int width, int height, int c_width, int c_height, float *zhol);

