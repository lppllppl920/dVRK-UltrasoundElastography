/*
 * Elastography.h
 *
 *  Created on: Mar 29, 2017
 *      Author: xingtong
 */

#ifndef SRC_ELASTOGRAPHY_H_
#define SRC_ELASTOGRAPHY_H_

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h> //inet_addr
#include <netdb.h>
#include <pthread.h> //for threading
#include <iostream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/progress.hpp>

#include "igtlOSUtil.h"
#include "igtlMessageHeader.h"
#include "igtlImageMessage.h"
#include "igtlImageMetaMessage.h"
#include "igtlLabelMetaMessage.h"
#include "igtlServerSocket.h"
#include "igtlStringMessage.h"
#include "igtlMUSMessage.h"

#include "elastography/concurrent_queue.h"
#include "elastography/scan_conversion.h"
#include "elastography/matrix_io.h"
#include "elastography/ncc.h"

class Elastography {
public:

    typedef unsigned char data_type_out;
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

    typedef struct tagRxCommand {
        /// Commands of Network Protocol
        /// 1: The request of connection is accepted
        /// 2: start of Data Transfer
        /// 3: US Data Transfer
        /// 4: stop of Data Transfer

        unsigned char command;

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
        float spacing[3];
        FrameHeader fhr;
        igtl::USMessage::Pointer ImgMsg;
    };

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

    Elastography(int argc, char** argv);
    virtual ~Elastography();
    void CalculateElastography();
    void PushRFData(void *image_data, int image_size, int image_height,
            int image_width, float image_space_0, float image_space_1,
            float image_space_2, int LineDensity, long SamplingFrequency,
            long TransmitFrequency, int FPS, time_t time_stamp);

private:

    void ReadRFData(std::string prefix, int count);
    void execute_TRuE(cost *C, vector<data_frame_queue *> *vec_a,
            data_frame_queue *rf_data, int height, int width, int no_of_frames,
            int top_n, int iteration_count, int overall_iteration_count,
            char* output_folder_name, FrameHeader fhr,
            int strain_or_displacement, int window, float overlap,
            float displacement);
    int ncc_thread_collector(ncc_parameters *ncc_p,
            ncc_collector_p data_collector, boost::thread workThread[],
            int index, int total, int overall_count);
    void wait_for_threads(boost::thread workThread[], int total);
    void perform_strain_average(unsigned char *avg_scaled_out,
            ncc_collector_p data_collector, int total);
    void free_collector(ncc_collector_p data_collector, int total);
    void calculate_true_cost(std::vector<data_frame_queue *> *vec_a, cost C[],
            int vector_size, int &count_el, int &get_pos, double effAx);
    void copy_float_double(double a[16], igtl::Matrix4x4 b);
    double EstimateCorr(const double Tr1[16], const double Tr2[16],
            const int ROIrect[4], const double ScaleXY[2], const double effAx,
            const double Sig[3]);
    float calculate_distance_probe(const igtl::Matrix4x4 &a,
            const igtl::Matrix4x4 &b);
    void read_burst_data(int is_burst, char **data,
            concurrent_queue<data_frame_queue *> &in_queue, int &height,
            int &width, double &iTime, FrameHeader &fhr);
    void divide_initial_data(short int *data, int height, int width,
            int burst_count);
    void add_char_data(short int *data, short int *recurring_data, int height,
            int width, int burst_count);
    bool gluInvertMatrix(const double m[16], double invOut[16]);
    void MulMatrices(const double A[16], const double B[16], double AB[16]);
    void GetDis(const double Tr1[16], const double Tr2[16],
            const int ROIrect[4], const double ScaleXY[2], double outputDD[3]);
    int thread_counter(boost::thread workThread[], int x, int y,
            boost::timer t1, FILE *fp);
    int thread_logger(boost::thread workThread[], double rf_time, FILE *fp);
    int ncc_thread_dummy(ncc_parameters *ncc_p);

private:
    long rf_count_;
    long iteration_count_;
    long overall_iteration_count_;
    float average_cross_;
    float noise_percentage_;
    double time_;
    float average_strain_;
    int height_;
    int width_;
    int scale_width_;
    int scale_height_;
    int strain_height_;
    int device_id_;
    int no_of_frames_;
    int top_n_;
    int strain_or_displacement_;
    int is_burst_;
    int vector_size_;
    int step_size_;
    int ncc_window_;
    float ncc_overlap_;
    float ncc_displacement_;
    // pre-compressed image and post-compressed image
    char *uncomp_;
    char *comp_;
    char *temp_;
    float *displacement_strain_;
    float *cross_corr_;
    unsigned char *strain_;
    unsigned char *average_output_strain_;
    unsigned char *network_average_strain_;
    unsigned char *scaled_strain_img_;

    boost::timer timer1_, timer2_;
    strain_out s_out_;
    FrameHeader fhr_;
    cost *costs_;
    std::vector<data_frame_queue *> *vec_rf_data_;
    data_frame_queue *rf_data_;
    concurrent_queue<data_frame_queue *> in_queue_;
};

#endif /* SRC_ELASTOGRAPHY_H_ */
