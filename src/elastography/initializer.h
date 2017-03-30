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

#ifndef _INITIALIZER_H
#define	_INITIALIZER_H

#ifdef _MSC_VER
#ifndef _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC
#endif
#include <crtdbg.h>
#else
#define _ASSERT(expr) ((void)0)
#define _ASSERTE(expr) ((void)0)
#endif

#define RECORD_TIME_IN_FILE

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <MUSiiCTCPServer.h>
#include <MUSiiCIGTLUtil.h>
#include <boost/progress.hpp>

#include "elastography/concurrent_queue.h"
#include "elastography/reader.h"
#include "elastography/TrUE_Corr.h"
#include "elastography/scan_conversion.h"
#include "elastography/matrix_io.h"
#include "elastography/ncc_thread.h"

#ifdef	__cplusplus
extern "C" {

float CROSSCORRELATION_THRESHOLD = (float) 0.65;
float NEGATIVE_THRESHOLD_STD_DEVIATION = (float) 3.0;
float NEGATIVE_THRESHOLD_CONSTANT = (float) 0.0;
float POSITIVE_THRESHOLD_STD_DEVIATION = (float) 2.0;
float POSITIVE_THRESHOLD_CONSTANT = (float) 0.0;
float STRAIN_VALUE_NEGATIVE_NOISE = (float) 0.005;
float STRAIN_VALUE_POSITIVE_NOISE = (float) 0.005;

#define PRINT_EI_TO_FILE 1
#endif

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

struct strain_out {
	int width;
	int height;
};

struct cost {
	int i;
	int j;
	float cost_f;
	float distance;
};

typedef strain_out *strain_out_p;

void execute_TRuE(cost *C, vector<data_frame_queue *> *vec_a,
		data_frame_queue *rf_data, int height, int width, int no_of_frames,
		int top_n, int iteration_count, int overall_iteration_count);
int print_matrix(FILE *fp, unsigned char *target, int x, int y, int N);
int print_matrix_float(FILE *fp, float *target, int x, int y, int N);
int print_matrix_short_int(FILE *fp, short int *target, int x, int y, int N);
void set_online_parameters(char *input);
void divide_initial_data(short int *data, int height, int width,
		int burst_count);
void add_char_data(short int *data, short int *recurring_data, int height,
		int width, int burst_count);
void add_strain_data(unsigned char *data, unsigned char *recurring_data,
		int height, int width, int max_iteration);
void copy_strain_data(unsigned char *data, unsigned char *source_data,
		int height, int width);
void read_burst_data(int is_burst, char **data,
		concurrent_queue<data_frame_queue *> &in_queue,
		concurrent_queue<data_frame_queue *> &out_queue, int &height,
		int &width, double &iTime, Probe &prb, FrameHeader &fhr);
void copy_float_double(double a[16], igtl::Matrix4x4 b);
float calculate_distance_probe(const igtl::Matrix4x4 &a,
		const igtl::Matrix4x4 &b);
void calculate_true_cost(vector<data_frame_queue *> *vec_a, cost C[],
		int vector_size, int &count_el, int &get_pos, double effAx = 0.4);
int ReceiveMsg(int numOfRun = 0, int taskInfo = 0, void* ptr = NULL,
		igtl::MessageBase::Pointer data1 = NULL, void* data2 = NULL,
		void* data3 = NULL);
void execute_TRuE(cost *C, vector<data_frame_queue *> *vec_a,
		data_frame_queue *rf_data, int height, int width, int no_of_frames,
		int top_n, int iteration_count, int overall_iteration_count);

FrameHeader fhr;
using boost::asio::ip::tcp;
using boost::timer;
using namespace std;
char *input_folder_name;
char *output_folder_name;
concurrent_queue<data_frame_queue *> in_queue;
concurrent_queue<data_frame_queue *> out_queue;
int drange_rf[2];
int drange_a[2];
float w_smooth;
float mu;
int window;
float displacement;
float overlap;
int lookahead;
int algorithm_choice;
int strain_or_displacement;
float crosscorrelation_threshold;
float strain_val_pos_noise;
float strain_val_neg_noise;
float positive_threshold_const;
float positive_threshold_std_dev;
float negative_threshold_const;
float negative_threshold_std_dev;
int is_burst;
int use_kal_lsqe;

#ifdef	__cplusplus
}
#endif

#endif	/* _INITIALIZER_H */

