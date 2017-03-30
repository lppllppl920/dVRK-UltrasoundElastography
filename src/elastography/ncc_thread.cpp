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

#include "ncc_thread.h"
extern "C" {
void ncc_slow(int height, int width, short int * comp_matrix_short,
		short int *uncomp_matrix_short, float **cross_corr,
		float **displacement, unsigned char **out_strain, int *out_height,
		float *average_cross, float *average_strain, float *noise_percentage,
		float F0, float FS, int strain_or_displacement, int NOF, int ncc_window,
		float ncc_overlap, float ncc_displacement);

void set_threshold_values(float crosscorrelation_threshold,
		float negative_threshold_std_deviation,
		float negative_threshold_constant,
		float positive_threshold_std_deviation,
		float positive_threshold_constant, float strain_value_negative_noise,
		float strain_value_positive_noise);
void cuda_scale_image_mm(data_type_out **out_Im1, int *out_width,
		int *out_height, data_type_out *Im1, int width, int height,
		int no_of_frames, double spacing_x, double spacing_y, float &time1);
}
int ncc_thread_dummy(ncc_parameters *ncc_p) {
	free(ncc_p->uncomp);
	free(ncc_p->comp);
	delete ncc_p;
	return 0;
}

int thread_logger(boost::thread workThread[], double rf_time, FILE *fp) {
	for (int i = 0; i < 100; i++)
		workThread[i].join();
	time_t now;
	time(&now);
	fprintf(fp, "%f %ld\n", rf_time, now);
	fflush(fp);
	return 0;
}

int thread_counter(boost::thread workThread[], int x, int y, boost::timer t1,
		FILE *fp) {

	for (int i = 0; i < 1000; i++)
		workThread[i].join();
	fprintf(fp, "X %d Y %d Time %lf\n", x, y, t1.elapsed());
	fflush(fp);
	return 0;
}

////////////////////////////////////////
int ncc_thread2(int NumOfRun, int TaskInof, void* ptr,
		igtl::MessageBase::Pointer data1, void* data2, void* data3) {
	ncc_parameters* ncc_p = (ncc_parameters*) data2;

	float *displacement_strain;
	float *cross_corr;
	unsigned char *strain;
	int strain_height;
	float average_cross;
	float average_strain;
	float noise_percentage;
	float scaling_time;

	//parameters for scan conversion code
	unsigned char *scaled_out;
	int scale_width;
	int scale_height;
	int scale_size;
	igtl::EIMessage::Pointer ImgMsg;

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

	for (int k = 0; k < ncc_p->no_of_frames; k++) {

		scale_size = scale_width * scale_height * sizeof(unsigned char);

		ImgMsg = igtl::EIMessage::New();

		igtl::Matrix4x4 temp_matrix;

		ImgMsg->SetScalarTypeToUint8();
		ImgMsg->SetDimensions(scale_width, scale_height, 1);
		ImgMsg->SetDeviceType("IMAGE");
		ImgMsg->SetDeviceName("EI_NCC");

		ImgMsg->SetDimensions(scale_width, scale_height, 1);

		ImgMsg->SetSpacing(ncc_p->spacing[0], ncc_p->spacing[1],
				ncc_p->spacing[2]);

		ImgMsg->AllocateScalars();

		memcpy(ImgMsg->GetScalarPointer(),
				scaled_out + k * scale_width * scale_height, scale_size);

		ncc_p->Original_ImgMsg->GetMatrix(temp_matrix);

		ImgMsg->SetMatrix(temp_matrix);

		ImgMsg->Pack();

		ncc_p->pServer->PutIGTLMessage((igtl::MessageBase::Pointer) ImgMsg);
	}

	free(ncc_p->comp);
	free(ncc_p->uncomp);
	free(scaled_out);
	delete ncc_p;
	return 1;
}

void wait_for_threads(boost::thread workThread[], int total) {
	int i;

	for (i = 0; i < total; i++) {
		workThread[i].join();
	}
}

void free_collector(ncc_collector_p data_collector, int total) {
	int i;

	for (i = 0; i < total; i++) {
		free(data_collector[i].image);
	}
}

void perform_strain_average(unsigned char *avg_scaled_out,
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

	if (avg_scaled_out_float == NULL) {
		printf("Could not allocated memory\n");
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

int ncc_thread_collector(ncc_parameters *ncc_p, ncc_collector_p data_collector,
		boost::thread workThread[], int index, int total, char *output_folder,
		int overall_count) {

	float *displacement_strain;
	float *cross_corr;
	unsigned char *strain;
	int strain_height;
	float average_cross;
	float average_strain;
	float noise_percentage;
	float scaling_time;

	//parameters for scan conversion code
	unsigned char *scaled_out;
	int scale_width;
	int scale_height;
	int scale_size;
	igtl::EIMessage::Pointer ImgMsg;

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
		wait_for_threads(workThread, total - 1);
		unsigned char *avg_scaled_out;
		avg_scaled_out = (unsigned char *) malloc(
				sizeof(unsigned char) * scale_width * scale_height);

		if (avg_scaled_out == NULL) {
			printf("Error allocating memory\n");
			return -1;
		}

		perform_strain_average(avg_scaled_out, data_collector, total);
		scale_size = scale_width * scale_height * sizeof(unsigned char);

		ImgMsg = igtl::EIMessage::New();
		igtl::Matrix4x4 temp_matrix;
		ImgMsg->SetScalarTypeToUint8();
		ImgMsg->SetDimensions(scale_width, scale_height, 1);
		ImgMsg->SetDeviceType("IMAGE");
		ImgMsg->SetDeviceName("EI_NCC");
		ImgMsg->SetDimensions(scale_width, scale_height, 1);
		ImgMsg->SetSpacing(ncc_p->spacing[0], ncc_p->spacing[1],
				ncc_p->spacing[2]);
		ImgMsg->AllocateScalars();
		// avg_scaled_out is elastography image generated by normalized cross-correlation method
		memcpy(ImgMsg->GetScalarPointer(), avg_scaled_out, scale_size);
		// Set the origin and orientation matrix to that of the pre-compressed image
		ncc_p->Original_ImgMsg->GetMatrix(temp_matrix);
		ImgMsg->SetMatrix(temp_matrix);
		// Write image to output directory
		char igtl_file_name[200];
		sprintf(igtl_file_name, "%s/%05d.igtl", output_folder, overall_count);
		igtl::MUSiiCIGTLMsgFileIO::Pointer pFileIO =
				igtl::MUSiiCIGTLMsgFileIO::New();
		pFileIO->WriteSingleFile((igtl::MessageBase::Pointer) ImgMsg,
				igtl_file_name);
		//TODO: We need another way to transmit and display the image message
//        ImgMsg->Pack();
//
//        // Send Image message through TCP server
//        ncc_p->pServer->PutIGTLMessage((igtl::MessageBase::Pointer) ImgMsg);
		free(avg_scaled_out);
		free_collector(data_collector, total);
		delete data_collector;
	}

	free(ncc_p->comp);
	free(ncc_p->uncomp);
	delete ncc_p;
	return 0;
}

