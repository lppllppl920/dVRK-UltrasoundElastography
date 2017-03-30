/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int read_matrix(FILE *fp, float *target, int N);
int print_matrix(FILE *fp, float *target, int x, int y, int N);
int count_matrix(FILE *fp, int *x, int *y);
int print_matrix_3(FILE *fp, short int *target, int height, int width,
		int N_width);
int print_matrix_4(int iteration_count, unsigned char *source, int height,
		int width, int N_width);
void scale_image_mm(unsigned char **out_Im1, int *out_width, int *out_height,
		unsigned char *Im1, int width, int height, int NOF, double spacing_x,
		double spacing_y);
