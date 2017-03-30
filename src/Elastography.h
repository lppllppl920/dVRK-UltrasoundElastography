/*
 * Elastography.h
 *
 *  Created on: Mar 29, 2017
 *      Author: xingtong
 */

#ifndef SRC_ELASTOGRAPHY_H_
#define SRC_ELASTOGRAPHY_H_

#include "elastography/initializer.h"
#define RECORD_TIME_IN_FILE

class Elastography {
public:
	Elastography(int argc, char** argv);
	virtual ~Elastography();

private:
	long iteration_count_;
	long overall_iteration_count_;
	Probe prb_;
	float average_cross_;
	float noise_percentage_;

	buffer input_;
	buffer output_;

	int prev_height_;
	int prev_width_;

	strain_out s_out_;
	double time_;

	float average_strain_;
	int height_;
	int width_;
	int strain_height_;

	int device_id_;
	int no_of_frames_;
	int top_n_;
	int rf_server_port_;
	int server_port_;

	int drange_rf_[2];
	int drange_a_[2];
	float w_smooth_;
	float mu_;
	int window_;
	float displacement_;
	float overlap_;
	int lookahead_;
	int algorithm_choice_;
	int strain_or_displacement_;
	float crosscorrelation_threshold_;
	float strain_val_pos_noise_;
	float strain_val_neg_noise_;
	float positive_threshold_const_;
	float positive_threshold_std_dev_;
	float negative_threshold_const_;
	float negative_threshold_std_dev_;
	int is_burst_;
	int use_kal_lsqe_;

	char *input_folder_name_;
	char *output_folder_name_;

	// uncompressed image and compressed image
	char *uncomp_;
	char *comp_;
	char *temp_;
	float *displacement_strain_;
	float *cross_corr_;
	unsigned char *strain_;
	unsigned char *average_output_strain_;
	unsigned char *network_average_strain_;
	unsigned char *scaled_strain_img_;

	FrameHeader fhr_;
	cost *costs_;
	vector<data_frame_queue *> *vec_rf_data_;
	data_frame_queue *rf_data_;
	concurrent_queue<data_frame_queue *> in_queue_;
	concurrent_queue<data_frame_queue *> out_queue_;
};

#endif /* SRC_ELASTOGRAPHY_H_ */
