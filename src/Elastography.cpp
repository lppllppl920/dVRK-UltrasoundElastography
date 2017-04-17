/*
 * Elastography.cpp
 *
 *  Created on: Mar 29, 2017
 *      Author: xingtong
 */

#include "Elastography.h"

void Elastography::ReadRFData(std::string prefix, int count) {

    data_frame_queue * rf_data_pointer;
    cv::Mat rf_image;
    for (int i = 0; i < count; i++) {
        rf_data_pointer = new data_frame_queue;
        std::cout << i << std::endl;
        std::stringstream directory;
        directory << prefix << "RF_" << i << ".tiff";
        std::cout << directory.str() << std::endl;
        rf_image = cv::imread(directory.str(), cv::IMREAD_GRAYSCALE);
//        cv::imshow("rf", rf_image);
//        cv::waitKey(100);
        std::cout << "Image size: "
                << rf_image.cols * rf_image.rows * rf_image.channels()
                << std::endl;
        rf_data_pointer->data = (char*) malloc(
                sizeof(unsigned char) * rf_image.cols * rf_image.rows
                        * rf_image.channels());
        memcpy(rf_data_pointer->data, rf_image.data,
                sizeof(unsigned char) * rf_image.cols * rf_image.rows
                        * rf_image.channels());
        rf_data_pointer->height = rf_image.rows;
        rf_data_pointer->width = rf_image.cols;
        rf_data_pointer->number_frames = 1;
        rf_data_pointer->itime = double(i);
        rf_data_pointer->spacing[0] = 0.111845;
        rf_data_pointer->spacing[1] = 0.112676;
        rf_data_pointer->spacing[2] = 1.0;
        //Redundant
        rf_data_pointer->ImgMsg = igtl::USMessage::New();
        rf_data_pointer->fhr = fhr_;
        in_queue_.push(rf_data_pointer);
    }
}

void Elastography::PushRFData(void *image_data, int image_size,
        int image_height, int image_width, float image_space_0,
        float image_space_1, float image_space_2, int LineDensity,
        long SamplingFrequency, long TransmitFrequency, int FPS,
        time_t time_stamp) {
    data_frame_queue * rf_data_pointer = new data_frame_queue;
    rf_data_pointer->data = (char*) malloc(image_size);
    memcpy(rf_data_pointer->data, image_data, image_size);
    rf_data_pointer->height = image_height;
    rf_data_pointer->width = image_width;
    rf_data_pointer->number_frames = 1;
    rf_data_pointer->itime = time_stamp;
    rf_data_pointer->spacing[0] = image_space_0;
    rf_data_pointer->spacing[1] = image_space_1;
    rf_data_pointer->spacing[2] = image_space_2;
    //Redundant
    rf_data_pointer->ImgMsg = igtl::USMessage::New();

    fhr_.sf = SamplingFrequency;
    fhr_.ld = LineDensity;
    fhr_.dr = FPS;
    fhr_.txf = TransmitFrequency;
    rf_data_pointer->fhr = fhr_;
    in_queue_.push(rf_data_pointer);
}

void Elastography::CalculateElastography() {

    // Read rf data from hard disk
//    ReadRFData("/home/xingtong/Pictures/US/", 30);

// Popping all the rf data we need to use to calculate the first frame of strain image
// vector_size_ is used to specify how many frames do we want to use to calculate a single strain image
    bool first_time = true;

    if (is_burst_ == 0) {

        for (int vector_loop = 0; vector_loop < vector_size_; vector_loop++) {
            std::cout << "Waiting to get rf data " << vector_loop << " ..."
                    << std::endl;
            in_queue_.wait_and_pop(rf_data_);
            uncomp_ = rf_data_->data;
            height_ = rf_data_->height;
            width_ = rf_data_->width;
            no_of_frames_ = rf_data_->number_frames;
            time_ = rf_data_->itime;
            fhr_ = rf_data_->fhr;
            set_threshold_values((float) fhr_.ss, (float) fhr_.uly,
                    (float) fhr_.ulx, (float) fhr_.ury, (float) fhr_.urx,
                    (float) fhr_.brx, (float) fhr_.bry);

            vec_rf_data_->push_back(rf_data_);
        }

    } else {
        read_burst_data(is_burst_, &uncomp_, in_queue_, height_, width_, time_,
                fhr_);
    }

    costs_ = (cost *) malloc(
            (vector_size_ * (vector_size_ - 1) / 2) * sizeof(cost));
    while (true) {
        // Get position of the first element of the valid element in the vector
        int get_pos = 0;
        // Count total permutations calculated
        int count_el = 0;
        int step_count = 0;

        if (is_burst_ == 0) {
#ifdef DEBUG_OUTPUT
            std::cout << "Waiting to get rf data\n";
#endif
            if (first_time) {
                first_time = false;
            } else {
                // Discard old frames of RF data and receive new frames
                while (step_count < step_size_) {
                    in_queue_.wait_and_pop(rf_data_);
                    uncomp_ = rf_data_->data;
                    height_ = rf_data_->height;
                    width_ = rf_data_->width;
                    no_of_frames_ = rf_data_->number_frames;
                    time_ = rf_data_->itime;
                    fhr_ = rf_data_->fhr;
                    set_threshold_values((float) fhr_.ss, (float) fhr_.uly,
                            (float) fhr_.ulx, (float) fhr_.ury,
                            (float) fhr_.urx, (float) fhr_.brx,
                            (float) fhr_.bry);

                    free((*vec_rf_data_)[0]->data);
                    vec_rf_data_->erase(vec_rf_data_->begin());
                    vec_rf_data_->insert(vec_rf_data_->end(), rf_data_);
                    step_count++;
                }
            }
            // Calculate the cost for every single pair of image
            calculate_true_cost(vec_rf_data_, costs_, vector_size_, count_el,
                    get_pos, 0.2);
        } else {
            read_burst_data(is_burst_, &uncomp_, in_queue_, height_, width_,
                    time_, fhr_);
        }

        std::cout << "Calculating... \n";
        execute_TRuE(costs_, vec_rf_data_, rf_data_, height_, width_,
                no_of_frames_, top_n_, iteration_count_,
                overall_iteration_count_, (char*) "", fhr_,
                strain_or_displacement_, ncc_window_, ncc_overlap_,
                ncc_displacement_);
        std::cout << "Calculating complete\n";
#ifdef DEBUG_OUTPUT
        std::cout << "strain image size: " << scale_height_ << " " << scale_width_ << std::endl;
#endif
        cv::Mat strain_image = cv::Mat::zeros(scale_height_, scale_width_,
        CV_8UC1);
        memcpy(strain_image.data, average_output_strain_,
                scale_width_ * scale_height_ * sizeof(unsigned char));
        free(average_output_strain_);

        cv::imshow("strain image", strain_image);
        cv::waitKey(100);
        iteration_count_++;
        fflush(stdout);
    }
}

Elastography::Elastography(int argc, char** argv) {

    if (argc != 10) {
        std::cout << "number of parameters is not right\n";
        return;
    }
    rf_count_ = 0;
    no_of_frames_ = 1;
    temp_ = NULL;
    average_output_strain_ = NULL;
    average_strain_ = 0.0;
    width_ = 0;
    displacement_strain_ = NULL;
    average_cross_ = 0.0;
    uncomp_ = NULL;
    comp_ = NULL;
    strain_ = NULL;
    network_average_strain_ = NULL;
    time_ = 0;
    height_ = 0;
    strain_height_ = 0;
    cross_corr_ = NULL;
    noise_percentage_ = 0;
    iteration_count_ = 0;
    strain_ = NULL;

    scaled_strain_img_ = NULL;
    costs_ = NULL;
    rf_data_ = NULL;
    vec_rf_data_ = new std::vector<data_frame_queue *>();
    overall_iteration_count_ = 0;

    fhr_.ss = 0.75 * 1000.0;
    fhr_.uly = 3 * 1000.0;
    fhr_.ulx = 1 * 1000.0;
    fhr_.ury = 2 * 1000.0;
    fhr_.urx = 0.0 * 1000.0;
    fhr_.brx = 0.035 * 10000.0;
    fhr_.bry = 0.035 * 10000.0;
    fhr_.txf = 5 * 1e6;
    fhr_.sf = 40 * 1e6;

    scale_height_ = 0;
    scale_width_ = 0;

    strain_or_displacement_ = atoi(argv[1]);
    is_burst_ = atoi(argv[2]);
    vector_size_ = atoi(argv[3]);
    device_id_ = atoi(argv[4]);
    top_n_ = atoi(argv[5]);
    step_size_ = atoi(argv[6]);
    ncc_window_ = atoi(argv[7]);//8;
    ncc_overlap_ = atof(argv[8]);//1.0;
    ncc_displacement_ = atof(argv[9]);//5.0;
// Set the desired cuda device
    set_cuda_device(device_id_);
}

void Elastography::execute_TRuE(cost *C, std::vector<data_frame_queue *> *vec_a,
        data_frame_queue *rf_data, int height, int width, int no_of_frames,
        int top_n, int iteration_count, int overall_iteration_count,
        char* output_folder_name, FrameHeader fhr, int strain_or_displacement,
        int window, float overlap, float displacement) {

    ncc_collector_p data_collector;
    ncc_parameters *ncc_p;

    data_collector = new ncc_collector[top_n];
    boost::thread *workerThread;

    // Creating top_n worker threads
    workerThread = new boost::thread[top_n];
    if (data_collector == NULL || workerThread == NULL) {
#ifdef DEBUG_OUTPUT
        printf("Out of memory\n");
#endif
        return;
    }

    for (int i = 0; i < top_n; i++) {
        ncc_p = new ncc_parameters();
        if (ncc_p == NULL) {
#ifdef DEBUG_OUTPUT
            printf("Out of memory\n");
#endif
            return;
        }

        // comp means compressed image
        // uncomp means uncompressed image
        int image_size = height * width * no_of_frames * sizeof(short int);
        ncc_p->height = height;
        ncc_p->width = width;
        ncc_p->no_of_frames = no_of_frames;
        ncc_p->comp = (short int *) malloc(image_size); //freed in ncc_thread
        ncc_p->uncomp = (short int *) malloc(image_size); //freed in ncc_thread
        if (ncc_p->comp == NULL || ncc_p->uncomp == NULL) {
#ifdef DEBUG_OUTPUT
            printf("Out of memory\n");
#endif
            return;
        }
        memcpy(ncc_p->comp, vec_a->at(C[i].i)->data, image_size);
        memcpy(ncc_p->uncomp, vec_a->at(C[i].j)->data, image_size);
        rf_data = vec_a->at(C[i].i);

        // F0 and FS are in MHz unit
        ncc_p->F0 = (float) (fhr.txf / 1e6);
        ncc_p->FS = (float) (fhr.sf / 1e6);
        ncc_p->strain_or_displacement = strain_or_displacement;
        ncc_p->no_of_frames = no_of_frames;
        ncc_p->NOF = no_of_frames;
        ncc_p->spacing[0] = rf_data->spacing[0];
        ncc_p->spacing[1] = rf_data->spacing[1];
        ncc_p->spacing[2] = rf_data->spacing[2];
        ncc_p->Original_ImgMsg = rf_data->ImgMsg;

        ncc_p->ncc_window = window;
        ncc_p->ncc_overlap = overlap;

        ncc_p->ncc_displacement = displacement;
        //}
        ncc_p->ss = fhr.ss;
        ncc_p->uly = fhr.uly;
        ncc_p->ulx = fhr.ulx;
        ncc_p->ury = fhr.ury;
        ncc_p->urx = fhr.urx;
        ncc_p->brx = fhr.brx;
        ncc_p->bry = fhr.bry;

        std::cout << "multi thread is working\n";
        workerThread[i] = boost::thread(&Elastography::ncc_thread_collector,
                this, ncc_p, data_collector, workerThread, i, top_n,
                overall_iteration_count);
        std::cout << "multi thread is complete\n";
    }
    std::cout << "Waiting to join\n";
    // Wait for all threads complete
    for (int i = 0; i < top_n; i++) {
        workerThread[i].join();
    }
    std::cout << "joining complete\n";
}

int Elastography::ncc_thread_collector(ncc_parameters *ncc_p,
        ncc_collector_p data_collector, boost::thread workThread[], int index,
        int total, int overall_count) {

    float *displacement_strain;
    float *cross_corr;
    unsigned char *strain;
    int strain_height;
    float average_cross;
    float average_strain;
    float noise_percentage;
    float scaling_time;

//parameters for scan conversion code
    unsigned char *avg_scaled_out;
    unsigned char *scaled_out;
    int scale_width;
    int scale_height;
    int scale_size;

    set_threshold_values((float) ncc_p->ss, (float) ncc_p->uly,
            (float) ncc_p->ulx, (float) ncc_p->ury, (float) ncc_p->urx,
            (float) ncc_p->brx, (float) ncc_p->bry);

    ncc_slow(ncc_p->height, ncc_p->width * ncc_p->no_of_frames, ncc_p->comp,
            ncc_p->uncomp, &cross_corr, &displacement_strain, &strain,
            &strain_height, &average_cross, &average_strain, &noise_percentage,
            ncc_p->F0, ncc_p->FS, ncc_p->strain_or_displacement, ncc_p->NOF,
            ncc_p->ncc_window, ncc_p->ncc_overlap, ncc_p->ncc_displacement);

    free(displacement_strain);

    free(cross_corr);

    data_collector[index].weight = average_strain;

    cuda_scale_image_mm(&scaled_out, &scale_width, &scale_height, strain,
            ncc_p->width, strain_height, ncc_p->no_of_frames, ncc_p->spacing[0],
            ncc_p->spacing[1] * ncc_p->height / (double) strain_height,
            scaling_time);

    /*scale_image_mm (&scaled_out, &scale_width, &scale_height,
     strain, ncc_p->width, strain_height, ncc_p->no_of_frames,
     ncc_p->spacing[0], ncc_p->spacing[1] * ncc_p->height / (double) strain_height);*/

    free(strain);

    ncc_p->spacing[1] = ncc_p->spacing[1] * ncc_p->height
            / (double) scale_height;
    int k = 0;

    data_collector[index].dims[0] = scale_width;
    data_collector[index].dims[1] = scale_height;
    data_collector[index].dims[2] = 1;
    data_collector[index].image = scaled_out;

    if (index == total - 1) {
        //wait for total-1 threads to finish, this is the last thread
        scale_width_ = scale_width;
        scale_height_ = scale_height;
        wait_for_threads(workThread, total - 1);
        average_output_strain_ = (unsigned char *) malloc(
                sizeof(unsigned char) * scale_width_ * scale_height_);
        perform_strain_average(average_output_strain_, data_collector, total);

        free_collector(data_collector, total);
        delete [] data_collector;
    }

    free(ncc_p->comp);
    free(ncc_p->uncomp);
    delete ncc_p;
    return 0;
}

void Elastography::wait_for_threads(boost::thread workThread[], int total) {
    int i;

    for (i = 0; i < total; i++) {
        workThread[i].join();
    }
}

void Elastography::perform_strain_average(unsigned char *avg_scaled_out,
        ncc_collector_p data_collector, int total) {
    int i, j, k;
    float total_weight = 0;
    int width;
    int height;
    float *avg_scaled_out_float;

    width = data_collector[0].dims[0];
    height = data_collector[0].dims[1];

    for (i = 0; i < total; i++) {
        total_weight += data_collector[i].weight;
    }
    avg_scaled_out_float = (float *) malloc(sizeof(float) * width * height);

    if (avg_scaled_out == NULL || avg_scaled_out_float == NULL) {
#ifdef DEBUG_OUTPUT
        printf("Could not allocated memory\n");
#endif
        exit(1);
    }

    memset(avg_scaled_out_float, 0, sizeof(float) * width * height);
    memset(avg_scaled_out, 0, sizeof(unsigned char) * width * height);

    for (i = 0; i < total; i++) {
        for (j = 0; j < height; j++) {
            for (k = 0; k < width; k++) {
                avg_scaled_out_float[j * width + k] += data_collector[i].weight
                        * data_collector[i].image[j * width + k];
            }
        }
    }

    for (j = 0; j < height; j++) {
        for (k = 0; k < width; k++) {
            avg_scaled_out_float[j * width + k] /= total_weight;
        }
    }

    for (j = 0; j < height; j++) {
        for (k = 0; k < width; k++) {
            avg_scaled_out[j * width + k] =
                    (unsigned char) avg_scaled_out_float[j * width + k];
        }
    }

    free(avg_scaled_out_float);

}

void Elastography::free_collector(ncc_collector_p data_collector, int total) {

    for (int i = 0; i < total; i++) {
        free(data_collector[i].image);
    }
}

void Elastography::calculate_true_cost(std::vector<data_frame_queue *> *vec_a,
        cost C[], int vector_size, int &count_el, int &get_pos, double effAx) {
    int ROIrect[4];
    int sz[3];
    float sp[3];
    double ScaleXY[2];
    cost temp;

    double trans1_d[16];
    double trans2_d[16];

    data_frame_queue *rf_data;

    std::vector<int>::reverse_iterator rit;

    count_el = 0;

    for (get_pos = 0; (rf_data = vec_a->at(get_pos)) == NULL; get_pos++) {

    }

// Calculate the cost
    for (int i = 0; i < vector_size - 1; i++) {

        for (int j = i + 1; j < vector_size; j++) {
            if (i == j) {
                continue;
            }

            igtl::Matrix4x4 trans1;
            igtl::Matrix4x4 trans2;
            rf_data = vec_a->at(i + get_pos);
            rf_data->ImgMsg->GetMatrix(trans1);
            copy_float_double(trans1_d, trans1);
            rf_data = vec_a->at(j + get_pos);
            rf_data->ImgMsg->GetMatrix(trans2);
            copy_float_double(trans2_d, trans2);
            rf_data->ImgMsg->GetDimensions(sz);
            rf_data->ImgMsg->GetSpacing(sp[0], sp[1], sp[2]);
            //rf_data->ImgMsg->GetSpacing(sp);
            ScaleXY[0] = sp[0];
            ScaleXY[1] = sp[1];
            //previous value 0.2
            double Sig[3] = { 0.2, 0.4, 0.2 };
            ROIrect[0] = -sz[0] / 2;
            ROIrect[1] = sz[0] / 2;
            ROIrect[2] = 0;
            ROIrect[3] = sz[1];
            C[count_el].cost_f = EstimateCorr((const double*) trans1_d,
                    (const double*) trans2_d, ROIrect, ScaleXY, effAx, Sig);
            C[count_el].i = i + get_pos;
            C[count_el].j = j + get_pos;
            C[count_el].distance = calculate_distance_probe(trans1, trans2);
            count_el++;
        }
    }

// Sorting the cost objects
    for (int i = 0; i < count_el - 1; i++) {
        for (int j = 0; j < count_el - 1; j++) {
            if (C[j].cost_f < C[j + 1].cost_f) {
                temp = C[j];
                C[j] = C[j + 1];
                C[j + 1] = temp;
            }
        }
    }
}

void Elastography::copy_float_double(double a[16], igtl::Matrix4x4 b) {

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            a[i * 4 + j] = (double) b[i][j];
        }
    }
}

float Elastography::calculate_distance_probe(const igtl::Matrix4x4 &a,
        const igtl::Matrix4x4 &b) {
    float diff_x;
    float diff_y;
    float diff_z;

    diff_x = a[0][3] - b[0][3];
    diff_y = a[1][3] - b[1][3];
    diff_z = a[2][3] - b[2][3];

    return sqrtf(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

}

void Elastography::read_burst_data(int is_burst, char **data,
        concurrent_queue<data_frame_queue *> &in_queue, int &height, int &width,
        double &iTime, FrameHeader &fhr) {
    data_frame_queue *rf_data;
    char *recurring_data;
    in_queue.wait_and_pop(rf_data);
    *data = rf_data->data;
    height = rf_data->height;
    width = rf_data->width;
    iTime = rf_data->itime;
    fhr = rf_data->fhr;
    divide_initial_data((short int *) *data, height, width, is_burst);
#ifdef DEBUG_OUTPUT
    printf("Read frame number: 1\n");
#endif
    set_threshold_values((float) fhr.ss, (float) fhr.uly, (float) fhr.ulx,
            (float) fhr.ury, (float) fhr.urx, (float) fhr.brx, (float) fhr.bry);
    for (int burst_loop = 1; burst_loop < is_burst; burst_loop++) {
        in_queue.wait_and_pop(rf_data);
        recurring_data = rf_data->data;
        add_char_data((short int *) *data, (short int *) recurring_data, height,
                width, is_burst);
#ifdef DEBUG_OUTPUT
        printf("Read frame number: %d\n", burst_loop + 1);
#endif
        iTime = rf_data->itime;
    }
}

void Elastography::divide_initial_data(short int *data, int height, int width,
        int burst_count) {
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            data[j + i * width] /= burst_count;
        }
    }
}

void Elastography::add_char_data(short int *data, short int *recurring_data,
        int height, int width, int burst_count) {
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            data[j + i * width] += recurring_data[j + i * width] / burst_count;
        }
    }
}

bool Elastography::gluInvertMatrix(const double m[16], double invOut[16]) {
    double inv[16], det;
    int i;

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

void Elastography::MulMatrices(const double A[16], const double B[16],
        double AB[16]) {
    double sum;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            sum = 0;
            for (int e = 0; e < 4; e++)
                sum += A[4 * i + e] * B[4 * e + j];
            AB[4 * i + j] = sum;
        }
}

void Elastography::GetDis(const double Tr1[16], const double Tr2[16],
        const int ROIrect[4], const double ScaleXY[2], double outputDD[3]) {
    double RelT[16];
    double invTr1[16];
// RelT = inv(Tr1)*Tr2;
    if (!gluInvertMatrix(Tr1, invTr1)) {
        invTr1[0] = invTr1[5] = invTr1[10] = invTr1[15] = 1.0;
        invTr1[1] = invTr1[2] = invTr1[3] = 0;
        invTr1[4] = invTr1[6] = invTr1[7] = 0;
        invTr1[8] = invTr1[9] = invTr1[11] = 0;
        invTr1[12] = invTr1[13] = invTr1[14] = 0;
    }
    MulMatrices(invTr1, Tr2, RelT);

    double nn[3], tt, theta;
//%% convert to axis-angle %%%
    tt = (RelT[0] + RelT[5] + RelT[10] - 1) / 2;
    if ((tt < 1) && (tt > -1)) // make sure acos returns real number
        theta = acos(tt);
    else
        theta = 0;

    nn[0] = RelT[9] - RelT[6];
    nn[1] = RelT[2] - RelT[8];
    nn[2] = RelT[4] - RelT[1];
    double norm_nn = sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
    for (int i = 0; i < 3; i++)
        nn[i] *= theta / norm_nn;
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    double X1 = ScaleXY[0] * ROIrect[0];
    double X2 = ScaleXY[0] * ROIrect[1];
    double Y1 = ScaleXY[1] * ROIrect[2];
    double Y2 = ScaleXY[1] * ROIrect[3];

// find RMS^2 of AvgD
// DD(1): -nn(3)*Y + RelT(1,4)
    if (fabs(nn[2]) > 0.000001)   // prevent round off error
        outputDD[0] =
                -1.0 / (3.0 * nn[2]) / (Y2 - Y1)
                        * (pow(-nn[2] * Y2 + RelT[3], 3)
                                - pow(-nn[2] * Y1 + RelT[3], 3));
    else
        outputDD[0] = RelT[3] * RelT[3];

// DD(2): nn(3)*X + RelT(2,4)
    if (fabs(nn[2]) > 0.000001)   // prevent round off error
        outputDD[1] = 1.0 / (3.0 * nn[2]) / (X2 - X1)
                * (pow(nn[2] * X2 + RelT[7], 3) - pow(nn[2] * X1 + RelT[7], 3));
    else
        outputDD[1] = RelT[7] * RelT[7];

// DD(3): nn(1)*Y - nn(2)*X + RelT(3,4)
    if (fabs(nn[0] * nn[1]) > 0.0000001)   // prevent round off error
        outputDD[2] = -1.0 / (12.0 * nn[0] * nn[1]) / (X2 - X1) / (Y2 - Y1)
                * (pow(nn[0] * Y2 - nn[1] * X2 + RelT[11], 4)
                        - pow(nn[0] * Y2 - nn[1] * X1 + RelT[11], 4)
                        - pow(nn[0] * Y1 - nn[1] * X2 + RelT[11], 4)
                        + pow(nn[0] * Y1 - nn[1] * X1 + RelT[11], 4));
    else if (fabs(nn[1]) > 0.000001)   // prevent round off error
        outputDD[2] = -1.0 / (3.0 * nn[1]) / (X2 - X1)
                * (pow(-nn[1] * X2 + RelT[11], 3)
                        - pow(-nn[1] * X1 + RelT[11], 3));
    else if (fabs(nn[0]) > 0.000001)   // prevent round off error
        outputDD[2] =
                1.0 / (3.0 * nn[0]) / (Y2 - Y1)
                        * (pow(nn[0] * Y2 + RelT[11], 3)
                                - pow(nn[0] * Y1 + RelT[11], 3));
    else
        outputDD[2] = RelT[11] * RelT[11];

}

double Elastography::EstimateCorr(const double Tr1[16], const double Tr2[16],
        const int ROIrect[4], const double ScaleXY[2], const double effAx,
        const double Sig[3]) {
    double DD[3];
    GetDis(Tr1, Tr2, ROIrect, ScaleXY, DD);
    double Crr = exp(
            -DD[0] / (4.0 * Sig[0] * Sig[0])
                    - 1.0 / (4.0 * Sig[1] * Sig[1])
                            * pow(fabs(DD[1] - effAx), 3) / (DD[1] + 0.0001)
                    - DD[2] / (4 * Sig[2] * Sig[2]));
    return Crr;
}

int Elastography::ncc_thread_dummy(ncc_parameters *ncc_p) {
    free(ncc_p->uncomp);
    free(ncc_p->comp);
    delete ncc_p;
    return 0;
}

int Elastography::thread_logger(boost::thread workThread[], double rf_time,
        FILE *fp) {
    for (int i = 0; i < 100; i++)
        workThread[i].join();
    time_t now;
    time(&now);
    fprintf(fp, "%f %ld\n", rf_time, now);
    fflush(fp);
    return 0;
}

int Elastography::thread_counter(boost::thread workThread[], int x, int y,
        boost::timer timer1, FILE *fp) {

    for (int i = 0; i < 1000; i++)
        workThread[i].join();
    fprintf(fp, "X %d Y %d Time %lf\n", x, y, timer1.elapsed());
    fflush(fp);
    return 0;
}

Elastography::~Elastography() {

}

