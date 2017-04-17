/***************************************************************************
 Copyright (c) 2014
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
 * File:   ncc.h
 * Author: ndeshmu1
 *
 * Created on November 18, 2009, 4:42 PM
 */

#ifndef _NCC_H
#define	_NCC_H

#include <stdio.h>
#include <stdlib.h>

//#include "cutil.h"    
//#include "cutil_inline_runtime.h"
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <driver_types.h>

#ifdef	__cplusplus
extern "C" {
#endif

//#define DEBUG_THRESHOLD_VALUES 1
void print_strain(FILE *fp, float* strain, int m, int n);
void print_image(FILE *fp, short int* image, int m, int n);

//#define PRINT_DEBUG_OUTPUT_NCC

enum {
	NCC_MAX_FRAME_SIZE = 1000,
	MEDIAN_FILT_WIDTH = 6,
	MEDIAN_FILT_HEIGHT = 9,
	MOVING_AVERAGE_DEPTH = 5,
	RF_THRESHOLD_FILTER = -25
};

enum {
	fire_x, fire_y, fire_z, fire_done
};

#ifndef NODE_DECLARATION
#define NODE_DECLARATION

struct node {
	int *data;
	int *transpose_data;
	void *next;
	void *prev;
	int m;
	int n;
};
typedef node *node_p;

struct buffer {
	int count;
	node_p head;
	node_p tail;
};
typedef buffer *buffer_p;

#endif

struct firing {
	int type;
	int position;
	int complete;
};

typedef firing *firing_p;

void set_threshold_values(float crosscorrelation_threshold,
		float negative_threshold_std_deviation,
		float negative_threshold_constant,
		float positive_threshold_std_deviation,
		float positive_threshold_constant, float strain_value_negative_noise,
		float strain_value_positive_noise);

__device__ float norm_mean(short int* input, int size);
__device__ float norm_mean_2(short int* input, int start, int size);
__device__ void normalizecrosscorr_2(short int *comp, short int *uncomp,
		int comp_x, int uncomp_x, float *gamma, float f_mean, float t_mean);
__device__ int max_index(float *a, int size);
__device__ float max_value(float *a, int size);

void count_sampl_pts(int uncomp_matrix_x, int uncomp_matrix_y, int window,
		float overlap, float displacement, int *x, float *Win_T, int *Win_S,
		float *Over_T, int *Over_S, float *max_strain, int *num_strain_points,
		int *correlation_size, float F0, float FS);

__global__ void normalizecrosscorr_jumbo(short int *comp_matrix,
		short int *uncomp_matrix, float *t, float *corr, float *EI_1);

void ncc_slow(int height, int width, short int * comp_matrix_short,
		short int *uncomp_matrix_short, float **cross_corr,
		float **displacement, unsigned char **out_strain, int *out_height,
		float *average_cross, float *average_strain, float *noise_percentage,
		float F0, float FS, int strain_or_displacement, int NOF, int ncc_window,
		float ncc_overlap, float ncc_displacement);

void cuda_malloc_host(void **ptr, size_t ptr_size);

void initialize_ncc_slow(int window, float overlap, float displacement);
void cuda_free_local(void *ptr, char *name);
int set_cuda_device(int device_id);
void cuda_copy_host(void *dst, void *src, size_t ptr_size);

#ifdef	__cplusplus
}
#endif

#endif	/* _NCC_H */

