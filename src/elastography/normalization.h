/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors

 Please see license.txt for further information.

 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DEBUG
__device__ float norm_mean(float* input, int size);
__device__ float norm_mean_2(float* input, int start, int size);
__device__ void normalizecrosscorr_2(float *comp, float *uncomp, int comp_x,
        int uncomp_x, float *gamma);
__device__ int max_index(float *a, int size);
__device__ float max_value(float *a, int size);
void count_sampl_pts(int uncomp_matrix_x, int uncomp_matrix_y, int window,
        float overlap, float displacement, int *x);
__global__ void normalizecrosscorr_jumbo(float *comp_matrix, int comp_matrix_x,
        int comp_matrix_y, float *uncomp_matrix, int uncomp_matrix_x,
        int uncomp_matrix_y, int window, float overlap, float displacement,
        float *t, float *corr, int *x, float *EI_1, int N, float *z);

