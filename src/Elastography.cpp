/*
 * Elastography.cpp
 *
 *  Created on: Mar 29, 2017
 *      Author: xingtong
 */

#include "Elastography.h"

Elastography::Elastography(int argc, char** argv) {

	if (argc != 33) {
		printf(
				"Usage:%s IP_address(RF_Server) DP_drange_rf_start DP_drange_rf_end DP_drange_a_start"
						" DP_drange_a_end DP_w_smooth DP_mu NCC_window NCC_displacement"
						" NCC_overlap lookahead algorithm_name use_kal_lsqe device_id is_burst strain_or_displacement server_port rf_server_port"
						"correlation_threshold neg_threshold_std_dev neg_threshold_const pos_threshold_std_dev pos_threshold_const"
						"strain_val_neg_noise strain_val_pos_noise tx_freq sampling_freq sequence_number vector_size top_n input_folder_name output_folder_name\n",
				argv[0]);
		return;
	}

	igtl::EIMessage::Pointer ImgMsg;
	ImgMsg = igtl::EIMessage::New();

//    igtl::MUSiiCTCPServer::Pointer pServer = igtl::MUSiiCTCPServer::New();
//    igtl::MUSiiCTCPClient::Pointer c = igtl::MUSiiCTCPClient::New();
//    c->AddExternalGlobalOutputCallbackFunction(ReceiveMsg,
//            "PostCallbackFunction");

	no_of_frames_ = 1;
	temp_ = NULL;
	prev_width_ = 0;
	average_output_strain_ = NULL;
	average_strain_ = 0.0;
	width_ = 0;
	displacement_strain_ = NULL;
	average_cross_ = 0.0;
	uncomp_ = NULL;
	comp_ = NULL;
	prev_height_ = 0;
	strain_ = NULL;
	network_average_strain_ = NULL;
	time_ = 0;
	height_ = 0;
	strain_height_ = 0;
	cross_corr_ = NULL;
	noise_percentage_ = 0;
	iteration_count_ = 0;

	crosscorrelation_threshold_ = 0.0;
	strain_ = NULL;
	strain_val_pos_noise_ = 0.0;
	strain_val_neg_noise_ = 0.0;
	negative_threshold_const_ = 0.0;
	negative_threshold_std_dev_ = 0.0;
	positive_threshold_const_ = 0.0;
	positive_threshold_std_dev_ = 0.0;

	server_port_ = atoi(argv[17]);
	strain_or_displacement_ = atoi(argv[16]);
	is_burst_ = atoi(argv[15]);
	algorithm_choice_ = atoi(argv[12]);
	rf_server_port_ = atoi(argv[18]);
	int vector_size = atoi(argv[29]);
	top_n_ = atoi(argv[30]);
	input_folder_name_ = argv[31];
	output_folder_name_ = argv[32];

	rf_data_ = NULL;
	vec_rf_data_ = new vector<data_frame_queue *>();
	////////////////////////////////////////////////////////////
	/*fhr_.ss = 0.75 * 1000.0;
	 fhr_.uly = 3 * 1000.0;
	 fhr_.ulx = 1 * 1000.0;
	 fhr_.ury = 2 * 1000.0;
	 fhr_.urx = 0.0 * 1000.0;
	 fhr_.brx = 0.035 * 10000.0;
	 fhr_.bry = 0.035 * 10000.0;
	 fhr_.txf = 5 * 1e6;
	 fhr_.sf = 40 * 1e6;*/

	fhr_.ss = atof(argv[19]) * 1000.0;
	fhr_.uly = atof(argv[20]) * 1000.0;
	fhr_.ulx = atof(argv[21]) * 1000.0;
	fhr_.ury = atof(argv[22]) * 1000.0;
	fhr_.urx = atof(argv[23]) * 1000.0;
	fhr_.brx = atof(argv[24]) * 10000.0;
	fhr_.bry = atof(argv[25]) * 10000.0;
	fhr_.txf = atof(argv[26]) * 1e6;
	fhr_.sf = atof(argv[27]) * 1e6;

//    c->ConnectToHost(argv[1], rf_server_port, "MD_US", true, true);

	device_id_ = atoi(argv[14]);
	set_cuda_device(device_id_);

	input_.count = 0;
	input_.head = NULL;
	input_.tail = NULL;
	output_.count = 0;
	output_.head = NULL;
	output_.tail = NULL;

	use_kal_lsqe_ = atoi(argv[13]);
	drange_rf_[0] = atoi(argv[2]);
	drange_rf_[1] = atoi(argv[3]);
	drange_a_[0] = atoi(argv[4]);
	drange_a_[1] = atoi(argv[5]);
	w_smooth_ = (float) atof(argv[6]);
	mu_ = (float) atof(argv[7]);
	window_ = atoi(argv[8]);
	displacement_ = (float) atof(argv[9]);
	overlap_ = (float) atof(argv[10]);
	lookahead_ = atoi(argv[11]);

//    pServer->CreateServer(server_port);

	costs_ = (cost *) malloc(
			(vector_size * (vector_size - 1) / 2 + 5) * sizeof(cost));
	overall_iteration_count_ = 0;

	timer t1, t2;
#ifdef RECORD_TIME_IN_FILE
	FILE *timer_record;
	// TODO: We need to change the path
	timer_record = fopen(
			"/home/xingtong/Projects/MySocket/src/my_socket/output/ncc_time",
			"a");

	if (timer_record == NULL) {
		printf("Error opening timer files, exiting\n");
		exit(1);
	}
#endif
	t1.restart();

#ifdef RECORD_TIME_IN_FILE
	boost::thread *workerThread;
#else
	boost::thread workerThread;
#endif

	char filenam1[200];
	strcpy(filenam1, input_folder_name_);
	boost::thread(read_directory1, filenam1);

#ifndef ONE_IMAGE
	if (is_burst_ == 0) {
		for (int vector_loop = 0; vector_loop < vector_size; vector_loop++) {
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
		read_burst_data(is_burst_, &uncomp_, in_queue_, out_queue_, height_,
				width_, time_, prb_, fhr_);
	}

	prev_height_ = height_;
	prev_width_ = width_;

	while (true) {

#ifdef DEBUG_OUTPUT
		sprintf (file_name, "/home/xingtong/Projects/MySocket/src/my_socket/output/comp_%d", iteration_count_);
		FILE* fp4 = fopen (file_name,"w");

		if (fp4 == NULL) {
			printf ("error opening file %s\n", strerror(errno));
			exit (EXIT_FAILURE);
		}
		print_matrix_short_int (fp4, (short int *)comp_, width_, height_, height_);
		fclose (fp4);
#endif // DEBUG_OUTPUT

#endif // ONE_IMAGE

		/**
		 * save the dimensions in case previous while loop is never executed
		 */
		prev_height_ = height_;
		prev_width_ = width_;
		int get_pos = 0; //get position of the first element of the valid element in the vector
		int count_el = 0; //count total permutations calculated

		if (is_burst_ == 0) {
			//TODO: When receiver obtain US Image, it should push the image into in_queue_
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

			out_queue_.push(vec_rf_data_->front()); //Push the top element of the vector back to the thread which
			//allocated data to it.
			vec_rf_data_->erase(vec_rf_data_->begin()); //remove the first element of the vector which we just passed back
			vec_rf_data_->insert(vec_rf_data_->end(), rf_data_); //insert the fresh element at the end of the vector

			t2.restart();

			calculate_true_cost(vec_rf_data_, costs_, vector_size, count_el,
					get_pos, 0.2); //calculate the True cost

			printf("Time to calculate TRuE: %lf\n", t2.elapsed());

		} else {
			read_burst_data(is_burst_, &uncomp_, in_queue_, out_queue_, height_,
					width_, time_, prb_, fhr_);
		}

//#ifndef ONE_IMAGE
//        //TODO: we need a way to transmit the calculated strain image to this class
//        execute_TRuE(costs_, vec_rf_data_, rf_data_, height_, width_,
//                no_of_frames_, top_n_, iteration_count_,
//                overall_iteration_count_);
//
//        iteration_count_++;
//        overall_iteration_count_++;
//        printf("ITERATION COUNT %d\n", overall_iteration_count_);
//        fflush(stdout);
//
//#ifdef RECORD_TIME_IN_FILE
//        if (iteration_count_ % 100 == 0) {
//            iteration_count_ = 0;
//            t1.restart();
//        }
//#endif
//
//        //TODO: ?????
//        continue;

		// TODO: comp_ and uncomp_ needs to be provided
		ncc_slow(height_, width_ * no_of_frames_, (short int *) comp_,
				(short int *) uncomp_, &cross_corr_, &displacement_strain_,
				&strain_, &strain_height_, &average_cross_, &average_strain_,
				&noise_percentage_, (float) (fhr_.txf / 1e6),
				(float) (fhr_.sf / 1e6), strain_or_displacement_,
				no_of_frames_);
		cout << "Elapsed time for NCC frame matching:" << t1.elapsed() << endl;
		char * out_string = (char *) malloc(sizeof(char) * 10000);

#ifdef DEBUG_OUTPUT
		sprintf (file_name, "/home/xingtong/Projects/MySocket/src/my_socket/output/displacement_%d.txt", iteration_count_);
		FILE *fp5 = fopen (file_name,"w");

		if (fp5 == NULL) {
			printf ("error opening file %s\n", strerror(errno));
			exit (EXIT_FAILURE);
		}

		print_matrix_float (fp5, displacement_strain_, width_, strain_height_, width_);
		fclose (fp5);

		sprintf (file_name, "/home/xingtong/Projects/MySocket/src/my_socket/output/strain_%d.txt", iteration_count_);
		FILE* fp = fopen (file_name,"w");

		if (fp == NULL) {
			printf ("error opening file %s\n", strerror(errno));
			exit (EXIT_FAILURE);
		}
		print_matrix (fp, strain_, width_, strain_height_, width_);
		fclose (fp);

		sprintf (file_name, "/home/xingtong/Projects/MySocket/src/my_socket/output/corr_%d.txt", iteration_count_);
		FILE *fp2 = fopen (file_name,"w");

		if (fp2 == NULL) {
			printf ("error opening file %s\n", strerror(errno));
			exit (EXIT_FAILURE);
		}
		print_matrix_float (fp2, cross_corr_, width_, strain_height_, width_);
		fclose (fp2);
#endif

		free(displacement_strain_);
#ifndef ONE_IMAGE
		//TODO: Why swapping this two images
		temp_ = comp_;
		comp_ = uncomp_;
		uncomp_ = temp_;
#endif

		s_out_.width = width_;
		s_out_.height = strain_height_;

		if (is_burst_ != 0) {
			if (iteration_count_ == 1) {
				//create a copy
				cuda_malloc_host((void **) &average_output_strain_,
						s_out_.width * s_out_.height * sizeof(unsigned char));
				cuda_malloc_host((void **) &network_average_strain_,
						s_out_.width * s_out_.height * sizeof(unsigned char));
				cuda_copy_host(average_output_strain_, strain_,
						s_out_.height * s_out_.width * sizeof(unsigned char));
				cuda_copy_host(network_average_strain_, strain_,
						s_out_.height * s_out_.width * sizeof(unsigned char));
			} else {
				//add the strain
				cuda_malloc_host((void **) &network_average_strain_,
						s_out_.width * s_out_.height * sizeof(unsigned char));
				add_strain_data(average_output_strain_, strain_, s_out_.height,
						s_out_.width, iteration_count_ + 1);
				cuda_copy_host(network_average_strain_, average_output_strain_,
						s_out_.height * s_out_.width * sizeof(unsigned char));
			}
		}

		//TODO: need to add client connected or not status message
//        if (pServer->IsConnectedClients()) {
		int size = s_out_.width * s_out_.height * sizeof(unsigned char);
		int scale_width;
		int scale_height;
		int scale_size;
		int out_x;
		int out_y;
		int out_z;
		float final_x_spacing;

		t1.restart();
		scale_image_mm(&scaled_strain_img_, &scale_width, &scale_height,
				strain_, s_out_.width, s_out_.height, no_of_frames_,
				rf_data_->spacing[0],
				rf_data_->spacing[1] * height_ / (double) s_out_.height);
		cout << "Elapsed time for scan conversion frame matching:"
				<< t1.elapsed() << endl;
		rf_data_->spacing[1] = rf_data_->spacing[1] * height_
				/ (double) scale_height;

		// Transmitting
		for (int k = 0; k < no_of_frames_; k++) {

			scale_size = scale_width * scale_height * sizeof(unsigned char);
			ImgMsg = igtl::EIMessage::New();
			igtl::Matrix4x4 temp_matrix;
			float *normals[3];
			ImgMsg->SetScalarTypeToUint8();
			ImgMsg->SetDeviceType("IMAGE");
			ImgMsg->SetDeviceName("EI_NCC");
			ImgMsg->SetDimensions(scale_width, scale_height, 1);
			ImgMsg->SetSpacing(rf_data_->spacing[0], rf_data_->spacing[1],
					rf_data_->spacing[2]);
			ImgMsg->AllocateScalars();
			memcpy(ImgMsg->GetScalarPointer(),
					scaled_strain_img_ + k * scale_width * scale_height,
					scale_size);

			rf_data_->ImgMsg->GetMatrix(temp_matrix);
			rf_data_->ImgMsg->GetNormals((float (*)[3]) normals);

			ImgMsg->SetMatrix(temp_matrix);
			ImgMsg->SetNormals((float (*)[3]) normals);

			ImgMsg->Pack();

			sprintf(out_string,
					"FRAME %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
					scale_width, scale_height, time_,
					scale_width * rf_data_->spacing[0],
					scale_height * rf_data_->spacing[1], 0, average_cross_,
					4 * rf_data_->spacing[1], average_strain_, 0);
			printf("%s", out_string);
			free(out_string);
			fflush (stdout);

#ifdef PRINT_EI_TO_FILE
			char file_name[80];
			//TODO: we need to change the file path
			sprintf(file_name, "c:\\ei_ncc\\disp_%ld", iteration_count_);
			FILE *fp = fopen(file_name, "w");
			print_matrix(fp,
					scaled_strain_img_ + k * scale_width * scale_height,
					scale_width, scale_height, scale_width);
			fclose(fp);
#endif
			iteration_count_++;
		}
		if (is_burst_ != 0) {
			printf("Image Size : %d\n", size);
		}
//        } else {
//            printf("No client connected\n");
//        }

		free(cross_corr_);

#ifndef ONE_IMAGE
		/**
		 * Send data back to the receiver thread
		 */
		if (is_burst_ == 0) {
			rf_data_->data = comp_;
			out_queue_.push(rf_data_);
		}
#endif

	} //while (true)

//    pServer->CloseServer();
	return;
}

Elastography::~Elastography() {

}

