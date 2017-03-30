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

#include "matrix_io.h"
//#include "c:/openIGT/Network/MusiicServerT.h"
#include "common.h"
#include <boost/progress.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <MUSiiCTCPServer.h>
#include <MUSiiCIGTLFileIO.h>
#include <MUSiiCIGTLUtil.h>
using boost::timer;

typedef struct {
	int height;
	int width;
	int no_of_frames;
	short int * comp;
	short int *uncomp;
	float F0;
	float FS;
	int strain_or_displacement;
	int NOF;
	float spacing[3];
	igtl::USMessage::Pointer Original_ImgMsg;
	igtl::MUSiiCTCPServer::Pointer pServer;
	int ncc_window;
	float ncc_overlap;
	float ncc_displacement;
	int ss;
	int uly;
	int ulx;
	int ury;
	int urx;
	int brx;
	int bry;
} ncc_parameters;

typedef struct {
	int dims[3];
	unsigned char* image;
	float weight;
} ncc_collector;

typedef ncc_collector * ncc_collector_p;

int ncc_thread_collector(ncc_parameters *ncc_p, ncc_collector_p data_collector,
		boost::thread workThread[], int index, int total, char *output_folder,
		int overall_count);
int thread_counter(boost::thread workThread[], int x, int y, boost::timer t1,
		FILE *fp);
int ncc_thread_dummy(ncc_parameters *ncc_p);
int ncc_thread(ncc_parameters *ncc_p);
int thread_logger(boost::thread workThread[], double rf_time, FILE *fp);
