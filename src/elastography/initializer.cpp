/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.
 ***************************************************************************/

/* 
 * File:   initializer.c
 * Author: ndeshmu1
 *
 * Created on November 15, 2009, 6:17 PM
 */

#include "elastography/initializer.h"

//#define DEBUG_OUTPUT

//10.162.34.139 -40 0 0 0 0.3 1 16 1 4 1 1

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int print_matrix(FILE *fp, unsigned char *target, int x, int y, int N) {
	int i, j;
	for (i = 0; i < y; i++) {
		//   printf("[%d]",i);
		for (j = 0; j < x; j++) {
			//printf("(%d,%f) ", j,*(target + i * N + j));
			fprintf(fp, "%d ", *(target + i * N + j));
		}
		fprintf(fp, "\n");
	}
	return 0;
}

int print_matrix_float(FILE *fp, float *target, int x, int y, int N) {
	int i, j;
	for (i = 0; i < y; i++) {
		//   printf("[%d]",i);
		for (j = 0; j < x; j++) {
			//printf("(%d,%f) ", j,*(target + i * N + j));
			fprintf(fp, "%.3f ", *(target + i * N + j));
		}
		fprintf(fp, "\n");
	}
	return 0;
}

int print_matrix_short_int(FILE *fp, short int *target, int x, int y, int N) {
	int i, j;
	for (j = 0; j < y; j++) {
		for (i = 0; i < x - 1; i++) {
			fprintf(fp, "%d ", *(target + i * N + j));
		}
		fprintf(fp, "%d", *(target + i * N + j));
		fprintf(fp, "\n");
	}
	return 0;
}

/*
 DP_drange_rf_start DP_drange_rf_end DP_drange_a_start
 
 DP_drange_a_end DP_w_smooth DP_mu NCC_window NCC_displacement
 
 NCC_overlap lookahead algorithm_name use_kal_lsqe device_id is_burst strain_or_displacement server_port rf_server_port
 
 correlation_threshold neg_threshold_std_dev neg_threshold_const pos_threshold_std_dev pos_threshold_const
 
 strain_val_neg_noise strain_val_pos_noise tx_freq sampling_freq sequence_number*/

void set_online_parameters(char *input) {
	char *token;

	token = strtok(input, " ");

	switch (atoi(token)) {
	case 1: //DP_Range_rf
		token = strtok(NULL, " ");
		drange_rf[0] = atoi(token);
		token = strtok(NULL, " ");
		drange_rf[1] = atoi(token);
		break;
	case 2: //DP_range_a
		token = strtok(NULL, " ");
		drange_a[0] = atoi(token);
		token = strtok(NULL, " ");
		drange_a[1] = atoi(token);
		break;
	case 3: //DP_w_smooth
		token = strtok(NULL, " ");
		w_smooth = atof(token);
		break;
	case 4: //DP_mu
		token = strtok(NULL, " ");
		mu = atof(token);
		break;
	case 5: //NCC_window_size
		token = strtok(NULL, " ");
		window = atoi(token);
		break;
	case 6: //NCC_displacement
		token = strtok(NULL, " ");
		displacement = atof(token);
		break;
	case 7: //NCC_overlap
		token = strtok(NULL, " ");
		overlap = atof(token);
		break;
	case 8: //lookahead
		token = strtok(NULL, " ");
		lookahead = atoi(token);
		break;
	case 9: //algorithm_name
		token = strtok(NULL, " ");
		algorithm_choice = atoi(token);
		break;
	case 10: //use_kal_lsqe
		token = strtok(NULL, " ");
		use_kal_lsqe = atoi(token);
		break;
	case 11: //is_burst
		token = strtok(NULL, " ");
		is_burst = atoi(token);
		break;
	case 12: //strain_or_displacement
		token = strtok(NULL, " ");
		strain_or_displacement = atoi(token);
		break;
	case 13: //correlation_threshold
		token = strtok(NULL, " ");
		crosscorrelation_threshold = atof(token);
		break;
	case 14: //negative_threshold_std_dev
		token = strtok(NULL, " ");
		negative_threshold_std_dev = atof(token);
		break;
	case 15: //neg_threshold_const
		token = strtok(NULL, " ");
		negative_threshold_const = atof(token);
		break;
	case 16: //positive_threshold_std_dev
		token = strtok(NULL, " ");
		positive_threshold_std_dev = atof(token);
		break;
	case 17: //positive_threshold_const
		token = strtok(NULL, " ");
		positive_threshold_const = atof(token);
		break;
	case 18: //strain_val_neg_noise
		token = strtok(NULL, " ");
		strain_val_neg_noise = atof(token);
		break;
	case 19: //strain_val_pos_noise
		token = strtok(NULL, " ");
		strain_val_pos_noise = atof(token);
		break;
	default:
		break;
	}

}

extern "C" {
void ncc_slow(int height, int width, short int * comp_matrix_short,
		short int *uncomp_matrix_short, float **cross_corr,
		float **displacement, unsigned char **out_strain, int *out_height,
		float *average_cross, float *average_strain, float *noise_percentage,
		float F0, float FS, int strain_or_displacement, int NOF);
void cuda_copy_host(void *dst, void *src, size_t ptr_size);
void dp_disparity(short int* image1_short_local, short int *image2_short_local,
		int image1_m, int image1_n, float w_smooth, float mu,
		unsigned char **strain_ret, int *out_height, float *average_strain,
		float *noise_percentage, int *drange_rf_local, int *drange_a_local,
		int strain_or_displacement);

void initialize_ncc_slow(int window, float overlap, float displacement);
void cuda_free_local(void *ptr, char *name);
void cuda_malloc_host(void **ptr, size_t ptr_size);
int set_cuda_device(int device_id);
void print_disparity(FILE *fp, float* disp, int m, int n);
void print_strain(FILE *fp, float* strain, int m, int n);
void set_threshold_values(float crosscorrelation_threshold,
		float negative_threshold_std_deviation,
		float negative_threshold_constant,
		float positive_threshold_std_deviation,
		float positive_threshold_constant, float strain_value_negative_noise,
		float strain_value_positive_noise);
}

//#define FILE_TESTING

void divide_initial_data(short int *data, int height, int width,
		int burst_count) {
	int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			data[j + i * width] /= burst_count;
		}
	}
}

void add_char_data(short int *data, short int *recurring_data, int height,
		int width, int burst_count) {
	int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			data[j + i * width] += recurring_data[j + i * width] / burst_count;
		}
	}
}

void add_strain_data(unsigned char *data, unsigned char *recurring_data,
		int height, int width, int max_iteration) {
	int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			data[j + i * width] *= (max_iteration - 1);
			data[j + i * width] += recurring_data[j + i * width];
			data[j + i * width] /= max_iteration;
		}
	}
}

void copy_strain_data(unsigned char *data, unsigned char *source_data,
		int height, int width) {
	int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			data[j + i * width] = source_data[j + i * width];
		}
	}
}
//read_burst_data (is_burst, &comp, in_queue, out_queue, &height, &width, &iTime, &prb, &fhr);
void read_burst_data(int is_burst, char **data,
		concurrent_queue<data_frame_queue *> &in_queue,
		concurrent_queue<data_frame_queue *> &out_queue, int &height,
		int &width, double &iTime, Probe &prb, FrameHeader &fhr) {
	data_frame_queue *rf_data;
	char *recurring_data;
	int burst_loop;
	in_queue.wait_and_pop(rf_data);
	*data = rf_data->data;
	height = rf_data->height;
	width = rf_data->width;
	iTime = rf_data->itime;
//	prb = rf_data->prb;	
	fhr = rf_data->fhr;
	divide_initial_data((short int *) *data, height, width, is_burst);
	printf("Read frame number: 1\n");
	set_threshold_values((float) fhr.ss, (float) fhr.uly, (float) fhr.ulx,
			(float) fhr.ury, (float) fhr.urx, (float) fhr.brx, (float) fhr.bry);
	for (burst_loop = 1; burst_loop < is_burst; burst_loop++) {
		in_queue.wait_and_pop(rf_data);
		recurring_data = rf_data->data;
		add_char_data((short int *) *data, (short int *) recurring_data, height,
				width, is_burst);
		printf("Read frame number: %d\n", burst_loop + 1);
		out_queue.push(rf_data);
		iTime = rf_data->itime;
	}
}

void copy_float_double(double a[16], igtl::Matrix4x4 b) {

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			a[i * 4 + j] = (double) b[i][j];
		}
	}
}

float calculate_distance_probe(const igtl::Matrix4x4 &a,
		const igtl::Matrix4x4 &b) {
	float diff_x;
	float diff_y;
	float diff_z;

	diff_x = a[0][3] - b[0][3];
	diff_y = a[1][3] - b[1][3];
	diff_z = a[2][3] - b[2][3];

	return sqrtf(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

}

void calculate_true_cost(vector<data_frame_queue *> *vec_a, cost C[],
		int vector_size, int &count_el, int &get_pos, double effAx = 0.4) {
	int ROIrect[4];
	int sz[3];
	float sp[3];
	double ScaleXY[2];
	cost temp;

	double trans1_d[16];
	double trans2_d[16];

	data_frame_queue *rf_data;

	vector<int>::reverse_iterator rit;

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

// Receive US Message
int ReceiveMsg(int numOfRun = 0, int taskInfo = 0, void* ptr = NULL,
		igtl::MessageBase::Pointer data1 = NULL, void* data2 = NULL,
		void* data3 = NULL);

void execute_TRuE(cost *C, vector<data_frame_queue *> *vec_a,
		data_frame_queue *rf_data, int height, int width, int no_of_frames,
		int top_n, int iteration_count, int overall_iteration_count) {

	ncc_collector_p data_collector;
	ncc_parameters *ncc_p;

	data_collector = new ncc_collector[top_n];
	boost::thread *workerThread;

	// Creating top_n worker threads
	workerThread = new boost::thread[top_n];
	if (data_collector == NULL || workerThread == NULL) {
		printf("Out of memory\n");
		return;
	}

	for (int i = 0; i < top_n; i++) {
		ncc_p = new ncc_parameters();
		if (ncc_p == NULL) {
			printf("Out of memory\n");
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
			printf("Out of memory\n");
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
		ncc_p->ss = fhr.ss;
		ncc_p->uly = fhr.uly;
		ncc_p->ulx = fhr.ulx;
		ncc_p->ury = fhr.ury;
		ncc_p->urx = fhr.urx;
		ncc_p->brx = fhr.brx;
		ncc_p->bry = fhr.bry;

#ifdef RECORD_TIME_IN_FILE

		workerThread[i] = boost::thread(ncc_thread_collector, ncc_p,
				data_collector, workerThread, i, top_n, output_folder_name,
				overall_iteration_count);

#else 
		workerThread = boost::thread (ncc_thread, ncc_p);
#endif

		printf("ITERATION COUNT %d\n", overall_iteration_count);
		fflush (stdout);
	}

	// Wait for all threads complete
	for (int i = 0; i < top_n; i++) {
		workerThread[i].join();
	}
}

//int main(int argc, char** argv) {
//
//#ifdef DEBUG_OUTPUT
//        char file_name[512];
//#endif
//        long iteration_count;
//        Probe prb;
//        float *displacement_strain;
//        float *cross_corr;
//        float average_cross;
//        float noise_percentage;
//
//        buffer input;
//        buffer output;
//
//        // uncompressed image and compressed image
//        char *uncomp;
//        char *comp;
//
//        char *temp;
//        int prev_height;
//        int prev_width;
//        unsigned char *strain;
//        unsigned char *average_output_strain;
//        unsigned char *network_average_strain;
//        strain_out s_out;
//        double iTime;
//
//        float average_strain;
//        int height;
//        int width;
//        int strain_height;
//
//        int device_id;
//        int no_of_frames = 1;
//        int top_n;
//        int rf_server_port;
//
//        igtl::EIMessage::Pointer ImgMsg;
//        ImgMsg = igtl::EIMessage::New();
//        igtl::MUSiiCTCPServer::Pointer pServer = igtl::MUSiiCTCPServer::New();
//
//        ////////////////////////////////////////////////////////////
//        igtl::MUSiiCTCPClient::Pointer c = igtl::MUSiiCTCPClient::New();
//        c->AddExternalGlobalOutputCallbackFunction(ReceiveMsg,
//                "PostCallbackFunction");
//
///*        boost::asio::io_service io_service_c;
//        boost::asio::io_service io_service;
//        boost::asio::io_service io_service_parameter_receiver;
//        tcp::endpoint endpoint(tcp::v4(), 27736);
//        tcp::endpoint endpoint_para_recv(tcp::v4(), 30000);
//        tcp::acceptor acceptor(io_service_parameter_receiver,
//                endpoint_para_recv); */
//
//        if (argc != 33) {
//            printf(
//                    "Usage:%s IP_address(RF_Server) DP_drange_rf_start DP_drange_rf_end DP_drange_a_start"
//                            " DP_drange_a_end DP_w_smooth DP_mu NCC_window NCC_displacement"
//                            " NCC_overlap lookahead algorithm_name use_kal_lsqe device_id is_burst strain_or_displacement server_port rf_server_port"
//                            "correlation_threshold neg_threshold_std_dev neg_threshold_const pos_threshold_std_dev pos_threshold_const"
//                            "strain_val_neg_noise strain_val_pos_noise tx_freq sampling_freq sequence_number vector_size top_n input_folder_name output_folder_name\n",
//                    argv[0]);
//            exit(EXIT_FAILURE);
//        }
//
//        SERVER_PORT = atoi(argv[17]);
//        strain_or_displacement = atoi(argv[16]);
//        is_burst = atoi(argv[15]);
//        algorithm_choice = atoi(argv[12]);
//        rf_server_port = atoi(argv[18]);
//        int vector_size = atoi(argv[29]);
//        top_n = atoi(argv[30]);
//        input_folder_name = argv[31];
//        output_folder_name = argv[32];
//
//        vector<data_frame_queue *> *vec_a = new vector<data_frame_queue *>();
//        ////////////////////////////////////////////////////////////
//        data_frame_queue *rf_data;
//
//        /*fhr.ss = 0.75 * 1000.0;
//         fhr.uly = 3 * 1000.0;
//         fhr.ulx = 1 * 1000.0;
//         fhr.ury = 2 * 1000.0;
//         fhr.urx = 0.0 * 1000.0;
//         fhr.brx = 0.035 * 10000.0;
//         fhr.bry = 0.035 * 10000.0;
//         fhr.txf = 5 * 1e6;
//         fhr.sf = 40 * 1e6;*/
//
//        fhr.ss = atof(argv[19]) * 1000.0;
//        fhr.uly = atof(argv[20]) * 1000.0;
//        fhr.ulx = atof(argv[21]) * 1000.0;
//        fhr.ury = atof(argv[22]) * 1000.0;
//        fhr.urx = atof(argv[23]) * 1000.0;
//        fhr.brx = atof(argv[24]) * 10000.0;
//        fhr.bry = atof(argv[25]) * 10000.0;
//        fhr.txf = atof(argv[26]) * 1e6;
//        fhr.sf = atof(argv[27]) * 1e6;
//
//        c->ConnectToHost(argv[1], rf_server_port, "MD_US", true, true);
//
//        device_id = atoi(argv[14]);
//        set_cuda_device(device_id);
//
//        input.count = 0;
//        input.head = NULL;
//        input.tail = NULL;
//        output.count = 0;
//        output.head = NULL;
//        output.tail = NULL;
//
//        use_kal_lsqe = atoi(argv[13]);
//        drange_rf[0] = atoi(argv[2]);
//        drange_rf[1] = atoi(argv[3]);
//        drange_a[0] = atoi(argv[4]);
//        drange_a[1] = atoi(argv[5]);
//        w_smooth = (float) atof(argv[6]);
//        mu = (float) atof(argv[7]);
//        window = atoi(argv[8]);
//        displacement = (float) atof(argv[9]);
//        overlap = (float) atof(argv[10]);
//        lookahead = atoi(argv[11]);
//
//        int server_port;
//        if (algorithm_choice == 1) {
//            if (use_kal_lsqe == 0) {
//                printf("Running 1: port %d\n", SERVER_PORT);
//            } else {
//                printf("Running 1 with Kalman filter: port %d\n", SERVER_PORT);
//            }
//            server_port = SERVER_PORT;
//        } else {
//            printf("Running 0: port %d\n", SERVER_PORT + 1);
//            server_port = SERVER_PORT + 1;
//        }
//
//        pServer->CreateServer(server_port);
//
//        cost *C = (cost *) malloc(
//                (vector_size * (vector_size - 1) / 2 + 5) * sizeof(cost));
//
//        iteration_count = 0;
//        int overall_iteration_count = 0;
//        timer t1, t2;
//
//#ifdef RECORD_TIME_IN_FILE
//        FILE *timer_record;
//        // TODO: We need to change the path
//        timer_record = fopen("/home/xingtong/Projects/Cmake_OTruesoftware/Elastography/output/ncc_time", "a");
//
//        if (timer_record == NULL) {
//            printf("Error opening timer files, exiting\n");
//            exit(1);
//        }
//#endif
//
//        t1.restart();
//
//#ifdef RECORD_TIME_IN_FILE
//        boost::thread *workerThread;
//#else
//        boost::thread workerThread;
//#endif
//
//        char filenam1[200];
//        strcpy(filenam1, input_folder_name);
//        boost::thread(read_directory1, filenam1);
//
//#ifndef ONE_IMAGE
//
//        if (is_burst == 0) {
//            for (int vector_loop = 0; vector_loop < vector_size;
//                    vector_loop++) {
//                in_queue.wait_and_pop(rf_data);
//                uncomp = rf_data->data;
//                height = rf_data->height;
//                width = rf_data->width;
//                no_of_frames = rf_data->number_frames;
//                iTime = rf_data->itime;
//                fhr = rf_data->fhr;
//                set_threshold_values((float) fhr.ss, (float) fhr.uly,
//                        (float) fhr.ulx, (float) fhr.ury, (float) fhr.urx,
//                        (float) fhr.brx, (float) fhr.bry);
//
//                vec_a->push_back(rf_data);
//            }
//
//        } else {
//            read_burst_data(is_burst, &uncomp, in_queue, out_queue, height,
//                    width, iTime, prb, fhr);
//        }
//
//        prev_height = height;
//        prev_width = width;
//
//        while (true) {
//
//#ifdef DEBUG_OUTPUT
//
//            sprintf (file_name, "/home/xingtong/Projects/Cmake_OTruesoftware/Elastography/output/comp_%d", iteration_count);
//            FILE* fp4 = fopen (file_name,"w");
//
//            if (fp4 == NULL) {
//                printf ("error opening file %s\n", strerror(errno));
//                exit (EXIT_FAILURE);
//            }
//            print_matrix_short_int (fp4, (short int *)comp, width, height, height);
//            fclose (fp4);
//#endif
//
//
//#endif
//
//            /**
//             * save the dimensions in case previous while loop is never executed
//             */
//            prev_height = height;
//            prev_width = width;
//            int get_pos = 0; //get position of the first element of the valid element in the vector
//            int count_el = 0; //count total permutations calculated
//            if (is_burst == 0) {
//                in_queue.wait_and_pop(rf_data);
//                uncomp = rf_data->data;
//                height = rf_data->height;
//                width = rf_data->width;
//                no_of_frames = rf_data->number_frames;
//                iTime = rf_data->itime;
//                fhr = rf_data->fhr;
//                set_threshold_values((float) fhr.ss, (float) fhr.uly,
//                        (float) fhr.ulx, (float) fhr.ury, (float) fhr.urx,
//                        (float) fhr.brx, (float) fhr.bry);
//
//                out_queue.push(vec_a->front()); //Push the top element of the vector back to the thread which
//                                                //allocated data to it.
//
//                vec_a->erase(vec_a->begin());//remove the first element of the vector which we just passed back
//
//                vec_a->insert(vec_a->end(), rf_data); //insert the fresh element at the end of the vector
//
//                t2.restart();
//
//                calculate_true_cost(vec_a, C, vector_size, count_el, get_pos,
//                        0.2); //calculate the True cost
//
//                printf("Time to calculate TRuE: %lf\n", t2.elapsed());
//
//            } else {
//                read_burst_data(is_burst, &uncomp, in_queue, out_queue, height,
//                        width, iTime, prb, fhr);
//            }
//
//#ifndef ONE_IMAGE
//
//            execute_TRuE(C, vec_a, pServer, rf_data, height, width,
//                    no_of_frames, top_n, iteration_count,
//                    overall_iteration_count);
//
//            /*calculate_true_cost (vec_a, C, vector_size, count_el, get_pos, 0.3); //calculate the True cost
//             execute_TRuE (C, vec_a, pServer,rf_data, height, width, no_of_frames, top_n, iteration_count,overall_iteration_count);
//
//             calculate_true_cost (vec_a, C, vector_size, count_el, get_pos, 0.4); //calculate the True cost
//             execute_TRuE (C, vec_a, pServer,rf_data, height, width, no_of_frames, top_n, iteration_count,overall_iteration_count);
//
//             calculate_true_cost (vec_a, C, vector_size, count_el, get_pos, 0.5); //calculate the True cost
//             execute_TRuE (C, vec_a, pServer,rf_data, height, width, no_of_frames, top_n, iteration_count,overall_iteration_count);
//
//             calculate_true_cost (vec_a, C, vector_size, count_el, get_pos, 0.6); //calculate the True cost
//             execute_TRuE (C, vec_a, pServer,rf_data, height, width, no_of_frames, top_n, iteration_count,overall_iteration_count);*/
//
//            iteration_count++;
//            overall_iteration_count++;
//            printf("ITERATION COUNT %d\n", overall_iteration_count);
//            fflush(stdout);
//
//#ifdef RECORD_TIME_IN_FILE
//            if (iteration_count % 100 == 0) {
//                iteration_count = 0;
//                t1.restart();
//            }
//#endif
//
//            continue;
//
//            switch (algorithm_choice) {
//            case 0:
//                //Cross-correlation method
//                ncc_slow(height, width * no_of_frames, (short int *) comp,
//                        (short int *) uncomp, &cross_corr,
//                        &displacement_strain, &strain, &strain_height,
//                        &average_cross, &average_strain, &noise_percentage,
//                        (float) (fhr.txf / 1e6), (float) (fhr.sf / 1e6),
//                        strain_or_displacement, no_of_frames);
//                cout << "Elapsed time for NCC frame matching:" << t1.elapsed()
//                        << endl;
//
//                break;
//            case 1:
//                //TODO: What happened to dynamic programming method??
//                //dp_disparity ((short int *)uncomp, (short int *)comp, height, width, w_smooth, mu, &strain,
//                //		&strain_height,  &average_strain, &noise_percentage, drange_rf, drange_a, strain_or_displacement);
//                //average_cross = 0;
//                cout << "Dynamic programming not supported yet\n";
//                break;
//            }
//
//            char * out_string = (char *) malloc(sizeof(char) * 10000);
//
//#ifdef DEBUG_OUTPUT
//            sprintf (file_name, "c:\\ei_out\\displacement_%d.txt", iteration_count);
//            FILE *fp5 = fopen (file_name,"w");
//
//            if (fp5 == NULL) {
//                printf ("error opening file %s\n", strerror(errno));
//                exit (EXIT_FAILURE);
//            }
//
//            print_matrix_float (fp5, displacement_strain, width, strain_height, width);
//            fclose (fp5);
//#endif
//
//            if (algorithm_choice == 0) {
//                //displacement_strain is now a local memory
//                //cuda_free_local (displacement_strain, "displacement_strain");
//                free(displacement_strain);
//            }
//
//#ifdef DEBUG_OUTPUT
//            sprintf (file_name, "c:\\ei_out\\strain_%d.txt", iteration_count);
//            FILE* fp = fopen (file_name,"w");
//
//            if (fp == NULL) {
//                printf ("error opening file %s\n", strerror(errno));
//                exit (EXIT_FAILURE);
//            }
//            print_matrix (fp, strain, width, strain_height, width);
//            fclose (fp);
//#endif
//
//
//#ifdef DEBUG_OUTPUT
//            sprintf (file_name, "c:\\ei_out\\corr_%d.txt", iteration_count);
//            FILE *fp2 = fopen (file_name,"w");
//
//            if (fp2 == NULL) {
//                printf ("error opening file %s\n", strerror(errno));
//                exit (EXIT_FAILURE);
//            }
//            print_matrix_float (fp2, cross_corr, width, strain_height, width);
//            fclose (fp2);
//#endif
//
//
//#ifndef ONE_IMAGE
//            temp = comp;
//            comp = uncomp;
//            uncomp = temp;
//#endif
//
//            s_out.width = width;
//            s_out.height = strain_height;
//
//            out_string[0] = '\0';
//            printf("%s\n", out_string);
//
//            if (is_burst != 0) {
//                if (iteration_count == 1) {
//                    //create a copy
//                    cuda_malloc_host((void **) &average_output_strain,
//                            s_out.width * s_out.height * sizeof(unsigned char));
//                    cuda_malloc_host((void **) &network_average_strain,
//                            s_out.width * s_out.height * sizeof(unsigned char));
//                    cuda_copy_host(average_output_strain, strain,
//                            s_out.height * s_out.width * sizeof(unsigned char));
//                    cuda_copy_host(network_average_strain, strain,
//                            s_out.height * s_out.width * sizeof(unsigned char));
//                } else {
//                    //add the strain
//                    cuda_malloc_host((void **) &network_average_strain,
//                            s_out.width * s_out.height * sizeof(unsigned char));
//                    add_strain_data(average_output_strain, strain, s_out.height,
//                            s_out.width, iteration_count + 1);
//                    cuda_copy_host(network_average_strain,
//                            average_output_strain,
//                            s_out.height * s_out.width * sizeof(unsigned char));
//                }
//            }
//
//            //TODO: need to add client connected or not status message
//            //if (pServer->IsConnected()) {
//            if (true) {
//                int size = s_out.width * s_out.height * sizeof(unsigned char);
//
//                unsigned char *scaled_out;
//                int scale_width;
//                int scale_height;
//                int scale_size;
//                int out_x;
//                int out_y;
//                int out_z;
//                float final_x_spacing;
//
//                // rf_data->spacing[1] = rf_data->spacing[1] * height / (double) s_out.height;
//
//                //The radius is 0 for microarray and scan angle is 60
//
//                t1.restart();
//
//                //scaled_out = cuda_curved_scan_convert_2(strain, &scale_width,
//                // &scale_height, &no_of_frames, s_out.width, s_out.height, no_of_frames, 0, 70, rf_data->spacing[1] * height / (double) s_out.height, &final_x_spacing);
//
//                // rf_data->spacing[1] = rf_data->spacing[1] * height / (double) s_out.height;
//                //rf_data->spacing[0] = final_x_spacing;
//
//                scale_image_mm(&scaled_out, &scale_width, &scale_height, strain,
//                        s_out.width, s_out.height, no_of_frames,
//                        rf_data->spacing[0],
//                        rf_data->spacing[1] * height / (double) s_out.height);
//
//                cout << "Elapsed time for scan conversion frame matching:"
//                        << t1.elapsed() << endl;
//
//                rf_data->spacing[1] = rf_data->spacing[1] * height
//                        / (double) scale_height;
//
//                for (int k = 0; k < no_of_frames; k++) {
//
//                    //scale_image_mm (&scaled_out, &scale_width, &scale_height,
////					 strain + k * size, s_out.width, s_out.height, rf_data->spacing[0], rf_data->spacing[1] * height / (double) s_out.height);
//
//                    scale_size = scale_width * scale_height
//                            * sizeof(unsigned char);
//
//                    ImgMsg = igtl::EIMessage::New();
//                    //  ImgMsg->SetDimensions(s_out.width, s_out.height, no_of_frames);
//                    igtl::Matrix4x4 temp_matrix;
//                    float *normals[3];
//                    ImgMsg->SetScalarTypeToUint8();
//                    ImgMsg->SetDimensions(scale_width, scale_height, 1);
//                    ImgMsg->SetDeviceType("IMAGE");
//                    ImgMsg->SetDeviceName("EI_NCC");
//
////			 ImgMsg->SetScalarTypeToUInt8();
//
//                    //ImgMsg->SetDimensions(s_out.width, s_out.height, no_of_frames);
//                    ImgMsg->SetDimensions(scale_width, scale_height, 1);
//
//                    ImgMsg->SetSpacing(rf_data->spacing[0], rf_data->spacing[1],
//                            rf_data->spacing[2]);
//                    //ImgMsg->SetSpacing (rf_data->spacing[0], rf_data->spacing[1] , rf_data->spacing[2]);
//
//                    ImgMsg->AllocateScalars();
//                    memcpy(ImgMsg->GetScalarPointer(),
//                            scaled_out + k * scale_width * scale_height,
//                            scale_size);
//
//                    rf_data->ImgMsg->GetMatrix(temp_matrix);
//                    rf_data->ImgMsg->GetNormals((float (*)[3]) normals);
//
//                    ImgMsg->SetMatrix(temp_matrix);
//                    ImgMsg->SetNormals((float (*)[3]) normals);
//
//                    ImgMsg->Pack();
////				if ( k == no_of_frames/2) {
//                    //   pServer->PutMessageData(ImgMsg);
//                    //}
//
//                    /*sprintf (out_string, "FRAME %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
//                     s_out.width, s_out.height, iTime, width*0.3, (height * 1540000.0/40000000.0), 0,
//                     average_cross, 4 * (height * 1540000.0/40000000.0) / height, average_strain, 0);*/
//
//                    sprintf(out_string,
//                            "FRAME %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
//                            scale_width, scale_height, iTime,
//                            scale_width * rf_data->spacing[0],
//                            scale_height * rf_data->spacing[1], 0,
//                            average_cross, 4 * rf_data->spacing[1],
//                            average_strain, 0);
//
//                    //s->Write(out_string, strlen(out_string));
//
//                    //s->Write(scaled_out + k * scale_width * scale_height, scale_size);
//                    printf("%s", out_string);
//                    free(out_string);
//                    fflush(stdout);
//
//#ifdef PRINT_EI_TO_FILE
//                    char file_name[80];
//                    //TODO: we need to change the file path
//                    sprintf(file_name, "c:\\ei_ncc\\disp_%ld", iteration_count);
//                    FILE *fp = fopen(file_name, "w");
//                    print_matrix(fp,
//                            scaled_out + k * scale_width * scale_height,
//                            scale_width, scale_height, scale_width);
//                    fclose(fp);
//#endif
//                    iteration_count++;
//                    //s->Write(out_string, strlen(out_string));
//                    /**
//                     * Send data out as binary stream
//                     *
//                     */
//                    //printf("Image Size : %d\n", size);
//                    //printf("Image Size : %d\n", size);
//                    //TODO replace this statement with out of strain data
//                    //s->Write(strain, size);
//                    /**
//                     * Send header out
//                     */
//                }
//                if (is_burst != 0) {
//                    printf("Image Size : %d\n", size);
//                    //TODO replace this statement with out of write data
//                    //s->Write(out_string, strlen(out_string));
//                    /**
//                     * Send data out as binary stream
//                     *
//                     */
//                    printf("Image Size : %d\n", size);
//                    //TODO replace this statement with out string data
//                    //s->Write(network_average_strain, size);
//                }
//            } else {
//                /**
//                 * Print error message that there is no client to whome data needs to be sent
//                 */
//                printf("No client connected\n");
//            }
//
//            if (algorithm_choice == 0) {
//                //   cuda_free_local (cross_corr, "cross_corr");
//                //This is now a local memory.
//                free(cross_corr);
//            }
//
//            /**
//             * delete the uncompressed image
//             */
//#ifndef ONE_IMAGE
//            /**
//             * Send data back to the receiver thread
//             */
//            if (is_burst == 0) {
//                rf_data->data = comp;
//                out_queue.push(rf_data);
//            }
//#endif
//
//        } //while (true)
//
//        pServer->CloseServer();
//        return (0);
//    } //main

//#endif

