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
 * File:   reader.h
 * Author: ndeshmu1
 *
 * Created on November 18, 2009, 5:49 PM
 */

#ifndef _READER_H
#define	_READER_H

//#include "SharedMemory.h"
#include "concurrent_queue.h"
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <deque>
#include <list>
#include <set>
#include <MUSiiCTCPClient.h>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>
#include <dirent.h>
#include <MUSiiCIGTLFileIO.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
typedef unsigned char BYTE;

//#include "c:/openIGT/Network/MusiicClient.h"
//#include "c:/openIGT/Network/MusiicClientT.h"

extern "C" {
void cuda_malloc_host(void **ptr, size_t ptr_size);
void cuda_free_local(void *ptr, char *name);
}

//static void ReceiveMsg (void *data, int s);

typedef struct tagRxCommand {
/// Commands of Network Protocol
/// 1: The request of connection is accepted
/// 2: start of Data Transfer
/// 3: US Data Transfer
/// 4: stop of Data Transfer

    BYTE command;

} RxCommand;

// taken from ulterius_def.h
typedef struct {

    /// data type - data types can also be determined by file extensions

    int type;

    /// number of frames in file

    int frames;

    /// width - number of vectors for raw data, image width for processed data

    int w;

    /// height - number of samples for raw data, image height for processed data

    int h;

    /// data sample size in bits

    int ss;

    /// roi - upper left (x)

    int ulx;

    /// roi - upper left (y)

    int uly;

    /// roi - upper right (x)

    int urx;

    /// roi - upper right (y)

    int ury;

    /// roi - bottom right (x)

    int brx;

    /// roi - bottom right (y)

    int bry;

    /// roi - bottom left (x)

    int blx;

    /// roi - bottom left (y)

    int bly;

    /// probe identifier - additional probe information can be found using this id

    int probe;

    /// transmit frequency

    int txf;

    /// sampling frequency

    int sf;

    /// data rate - frame rate or pulse repetition period in Doppler modes

    int dr;

    /// line density - can be used to calculate element spacing if pitch and native # elements is known

    int ld;

    /// extra information - ensemble for color RF

    int extra;

} FrameHeader;

typedef struct tagRxFrameHeader {
    /// The Frame Header of US Data
    FrameHeader frh;
    /// The Frame Tag of US Data
    double frametag;

} RxFrameHeader;

struct Probe {
    /// Name of probe    
    char name[80];
    /// ID of the probe. The code pins programmed into connector.
    int id;
    /// Type of probe. See the probeType enumeration.
    int type;
    /// Number of elements in the probe.
    int elements;
    /// Pitch, or the distance between elements in microns.
    int pitch;
    /// Radius of probe in 1/1000th of degree.
    int radius;
    /// Central frequency of probe in Hz.
    int centerFrequency;
    /// Frequency bandwidth in Hz
    int frequencyBandwidth;
    /// Transmit offset in number of elements
    int transmitoffset;
    /// Maximum steer angle allowable in 1/1000th of degree.
    int maxsteerangle;
    /// Maximum focus distance in microns for calculating dynamic receive time delays.
    int maxfocusdistance;
    /// Minimum line duration for one scanline in microseconds.
    int minlineduration;
    /// Minimum focus distance for one Doppler scanline.
    int minFocusDistanceDoppler;
    /// Pin offset if the probe element #0 starts at a different pin.
    unsigned char pinOffset;
    char unusedC;
    short unusedS;
    /// Field of view of the motor range in 1/1000th of degree.
    int motorFOV;
    /// Depth of entry for the biopsy needle in microns.
    int biopsyDepth;
    /// Angle of entry for the biopsy needle in 1/1000th of degree.
    int biopsyAngle;
    /// Initial target distance at probe surface in microns.
    int biopsyDistance;
    /// Probe options.
    int options;
    /// Maximum number of motor steps if the probe has a motor
    int motorSteps;
    /// Radius of motor.
    int motorRadius;
    /// If the motor makes audible noise, set this flag to correct it
    unsigned short motorMinTimeBetweenPulses;
    /// Code for how the motor gets to home position.
    char motorHomeMethod;
    /// Default width in pixels of the biopsy guide.
    unsigned char biopsyWidth;
};

struct data_frame_queue {
    char *data;
    int number_frames;
    int height;
    int width;
    double itime;
    //Probe prb;
    float spacing[3];
    FrameHeader fhr;
    igtl::USMessage::Pointer ImgMsg;
};

void initialize_shared_memory();

//void get_frame (char **data, int *height, int *width, SharedMemory &shMemClient, double *itime, Probe &prb, FrameHeader &fhr);
//bool get_frame (char **data, int *height, int *width, CMusiicClientT<igtl::USMessage, igtl::USMessage::Pointer>* pclient, double *itime, Probe &prb, FrameHeader &fhr);
//bool get_frame(char *data, int *height, int* width, SharedMemory &shMemClient, double *itime, Probe &prb, FrameHeader &fhr);

//	bool ReceiveRFDataCIM(const FrameHeader& frh, const double& tag, SharedMemory &shMemClient, char **data);

bool GetProbeData(int ProbeID, Probe& probe);

//bool ReadDataFromTCP(char* data, int sz, CMusiicClientT<igtl::USMessage, igtl::USMessage::Pointer>* pclient);
int read_directory1(const char * path_name);
/*
 * Reference http://blog.emptycrate.com/node/277
 */

#endif	/* _READER_H */

