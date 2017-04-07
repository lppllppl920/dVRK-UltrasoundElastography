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
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "concurrent_queue.h"
#include "reader.h"
#include "TrUE_Corr.h"

#include <MUSiiCTCPServer.h>
#include <MUSiiCIGTLUtil.h>

#include "scan_conversion.h"
#include "matrix_io.h"
//#define ONE_IMAGE 1
//#define _WIN32_WINNT 0x0501
//#include "ncc.h"
#include <boost/progress.hpp>
#include "ncc_thread.h"

//#include "multithreading.h"
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

    int SERVER_PORT;

#ifdef	__cplusplus
    }
#endif

#endif	/* _INITIALIZER_H */

