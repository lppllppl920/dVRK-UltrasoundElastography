/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern "C" {
__device__ float arr_mean(float* input, int size);
__device__ float find_Slope(float *x, float* y, int length);
__global__ void st_LSQSE(float *disp, float* strain, int Kernel, int N);
__global__ void st_clear(float* strain, int N);
}
