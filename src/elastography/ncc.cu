/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors

 Please see license.txt for further information.

 ***************************************************************************/
//#include "differentiation.h"
//#include "median_filter.h"
#include "ncc.h"
//#define PRINT_DEBUG

int global_file_count;
buffer_p ncc_input;
buffer_p ncc_output;

int ncc_start_frame;
int ncc_end_frame;
int ncc_lookahead;
int *ncc_frame_buffer;

float CROSSCORRELATION_THRESHOLD;
float NEGATIVE_THRESHOLD_STD_DEVIATION;
float NEGATIVE_THRESHOLD_CONSTANT;
float POSITIVE_THRESHOLD_STD_DEVIATION;
float POSITIVE_THRESHOLD_CONSTANT;
float STRAIN_VALUE_NEGATIVE_NOISE;
float STRAIN_VALUE_POSITIVE_NOISE;

__device__ __constant__ float CROSSCORRELATION_THRESHOLD_CUDA;
__device__ __constant__ float NEGATIVE_THRESHOLD_STD_DEVIATION_CUDA;
__device__ __constant__ float NEGATIVE_THRESHOLD_CONSTANT_CUDA;
__device__ __constant__ float POSITIVE_THRESHOLD_STD_DEVIATION_CUDA;
__device__ __constant__ float POSITIVE_THRESHOLD_CONSTANT_CUDA;
__device__ __constant__ float STRAIN_VALUE_NEGATIVE_NOISE_CUDA;
__device__ __constant__ float STRAIN_VALUE_POSITIVE_NOISE_CUDA;

__device__ __constant__ int uncomp_matrix_x_const;
__device__ __constant__ int uncomp_matrix_y_const;
__device__ __constant__ int Win_S_const;
__device__ __constant__ int Over_S_const;
__device__ __constant__ int N_const;
__device__ __constant__ float max_strain_const;
__device__ __constant__ int num_strain_points_const;
__device__ __constant__ int volume_size_const;

void allocate_constant(int uncomp_matrix_x, int uncomp_matrix_y, int N,
		int Win_S, int Over_S, float max_strain, int num_strain_points,
		int volume_size) {
	cudaMemcpyToSymbol(uncomp_matrix_x_const, &uncomp_matrix_x, sizeof(int), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(uncomp_matrix_y_const, &uncomp_matrix_y, sizeof(int), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(N_const, &N, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Win_S_const, &Win_S, sizeof(int), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Over_S_const, &Over_S, sizeof(int), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(max_strain_const, &max_strain, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(num_strain_points_const, &num_strain_points, sizeof(int),
			0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(volume_size_const, &volume_size, sizeof(int), 0,
			cudaMemcpyHostToDevice);
}

/*
 Derived from
 1. http://www.codecogs.com/reference/c/math.h/acos.php?alias=cacos
 2. http://en.wikipedia.org/wiki/Complex_logarithm
 3. http://mathworld.wolfram.com/ComplexArgument.html
 */

/*
 * picked up from Imaging book
 * http://homepages.inf.ed.ac.uk/rbf/BOOKS/PHILLIPS/cips2ed.pdf page 484
 */
__device__ float arr_mean(float* input, int size) {
    int i;
    float sum = 0;
    for (i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum / size;
}

__device__ void fswap(float *a, float *b) {
    float temp;
    temp = *a;
    *a = *b;
    *b = temp;
} /* ends swap */

/*
 * picked up from Imaging book
 * http://homepages.inf.ed.ac.uk/rbf/BOOKS/PHILLIPS/cips2ed.pdf page 484
 */
__device__ void fsort_elements(float elements[], int count) {
    int i, j;
    j = count;
    while (j-- > 1) {
        for (i = 0; i < j; i++) {
            if (elements[i] > elements[i + 1])
                fswap(&elements[i], &elements[i + 1]);
        }
    }
}

__device__ float find_Slope(float *x, float* y, int length) {
    int i;
    float x_mean, y_mean, slope;
    float a, numer, denom;

    x_mean = arr_mean(x, length);
    y_mean = arr_mean(y, length);

    numer = 0;
    denom = 0;

    for (i = 0; i < length; i++) {
        a = x[i] - x_mean;
        numer += a * (y[i] - y_mean);
        denom += a * a;
    }

    slope = numer / denom;
    return slope;
}

__global__  void st_clear(float* strain, int N) {

	int i = threadIdx.x;
	int j = blockIdx.x;

	N *= volume_size_const;

	*(strain + i * N + j) = 0;
}

__global__ void st_LSQSE(float *disp, float* strain, int Kernel, int N) {

	int i = threadIdx.x;
	int j = blockIdx.x;

	int index, offset;
	float x[20], y[20];

	N *= volume_size_const;

	offset = round((double) Kernel / 2);

	for (index = 0; index < Kernel; index++) {
		x[index] = i + index + 1;
		y[index] = *(disp + (i + index) * N + j);
	}
	*(strain + (i + offset) * N + j) = find_Slope(x, y, Kernel);

}

__global__ void moving_average(float *disp, float *target, int moving,
		int width) {

	int x = threadIdx.x;
	int y = blockIdx.x;

	float *temp;
	int i;
	float prev = 0;
	temp = (disp + width * x + y);

	width *= volume_size_const;

	for (i = 0; i < moving; i++) {
		if ((x - i) < 0) {

		} else {
			prev += *(temp - width * i);
		}
	}
	*(target + width * x + y) = (float) prev / moving;
}
__global__ void median_filter(float *disp, float *target, int w, int h,
		int width, int height, int c_width, int c_height, float *zhol) {

	int x = threadIdx.x;
	int y = blockIdx.x;

	float *temp;
	int max_median = w * h / 2;
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;

	float temp_temp[70];

	width *= volume_size_const;
	height *= volume_size_const;

	temp = (disp + width * x + y);

	for (i = (-h / 2); i <= h / 2; i++, k++) {
		l = 0;
		for (j = (-w / 2); j <= w / 2; j++, l++) {
			/*
			 * x and y are problems here since they will surpass the value of small limits :(
			 *
			 */
			if ((i + x) < 0 || (i + x) > (c_height - 1) || (j + y % c_width) < 0
					|| (j + y % c_width) > (c_width - 1)) {
				*(temp_temp + k * w + l) = 0.0;
			} else {
				*(temp_temp + k * w + l) = *(temp + i * width + j);
			}
		}
	}

	fsort_elements(temp_temp, w * h);

	if ((w * h) % 2 != 0) {
		//     *(target + width*x + y) = *temp;
		*(target + width * x + y) = temp_temp[max_median];
	} else {
		*(target + width * x + y) = (temp_temp[max_median]
				+ temp_temp[max_median - 1]) / 2;
	}
}

__device__ float complex_acos(float number) {
	float real;
	float imaginary;
	float absolute_value;
	if (number > 0) {
		real = 0;
	} else {
		real = 3.14159265358979323846;
	}
	imaginary = -log((fabsf(number) + sqrt(number * number - 1)));
	absolute_value = sqrt(real * real + imaginary * imaginary);
	///printf("%f+%f*:%f\n", real, imaginary, absolute_value);
	return absolute_value;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
}

void average_cross_correlation(float *cross_corr, int width, int height,
		float *average_cross) {
	int i;
	int j;
	int total = width * height;
	float addition = 0;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			addition += cross_corr[i * width + j];
		}
	}
	*average_cross = addition / total;
}

void find_min_max_displacement(float *displacement, int width, int height,
		float *disp_min, float *disp_max) {
	int i;
	int j;

	//negative_threshold = mean - (3.0 * std_deviation);
	*disp_min = 1000;
	*disp_max = -1;

	/**
	 * Loop for entire image
	 */
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			if (*disp_min > displacement[j * width + i]) {
				*disp_min = displacement[j * width + i];
			}
			if (*disp_max < displacement[j * width + i]) {
				*disp_max = displacement[j * width + i];
			}
		}
	}
}

/**
 * Find min and max values in the given image
 */
void find_min_max_dev(float *strain, int width, int height, float *strain_min,
		float *strain_max, float *std_deviation, float *strain_average,
		float *noise_percentage) {
	int i;
	int j;
	int threshold_count;
	int nan_count = 0;
	float sum = 0;
	float average;
	float temp;
	float negative_threshold;
	float positive_threshold;

	//negative_threshold = mean - (3.0 * std_deviation);
	*strain_min = 1000;
	*strain_max = -1;

	/**
	 * Loop for entire image
	 */
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
		    if(strain[j * width + i] != strain[j * width + i]) {

		        if (j >= 1) {
		            strain[j * width + i] = strain[(j - 1) * width + i];
		        } else if (i >= 1){
		            strain[j * width + i] = strain[j * width + i - 1];
		        } else {
		            strain[j * width + i] = 0.0;
		        }

		        nan_count++;
		    }
			sum += strain[j * width + i];
			if (*strain_min > strain[j * width + i]) {
				*strain_min = strain[j * width + i];
			}
			if (*strain_max < strain[j * width + i]) {
				*strain_max = strain[j * width + i];
			}
		}
	}

	average = sum / (width * height);
	*strain_average = average;

	sum = 0;

	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			temp = strain[j * width + i] - average;
			sum += temp * temp;
		}
	}

	*std_deviation = sqrtf(sum / (width * height - 1));

	if (POSITIVE_THRESHOLD_STD_DEVIATION > 0) {
		positive_threshold = (*strain_average)
				+ (float) POSITIVE_THRESHOLD_STD_DEVIATION * (*std_deviation);
	} else {
		positive_threshold = POSITIVE_THRESHOLD_CONSTANT;
	}

	if (NEGATIVE_THRESHOLD_STD_DEVIATION > 0) {
		negative_threshold = (*strain_average)
				- (float) NEGATIVE_THRESHOLD_STD_DEVIATION * (*std_deviation);
	} else {
		negative_threshold = NEGATIVE_THRESHOLD_CONSTANT;
	}

	threshold_count = 0;

	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			//printf ("width %d j %d i %d\n", width, j , i);
			//fflush(stdout);
			if (strain[j * width + i] >= positive_threshold
					|| strain[j * width + i] <= negative_threshold) {
				threshold_count++;
			}
		}
	}
	*noise_percentage = ((float) threshold_count / (height * width)) * 100;
}

/**
 * Function to map [0 1] to [0 255]
 */
void __global__ map_255_colormap(float *strain_d, unsigned char *strain_char_d,
		int width, int height, float strain_min, float diff) {
	/**
	 * Get the x and y coordinates of the images
	 */
	int x = blockIdx.x;
	int y = threadIdx.x;

	/**
	 * Do the conversion b  = (a * min / diff) * 255;
	 */
	strain_char_d[y * width + x] = (unsigned char) ((((strain_d[y * width + x]
			- strain_min) / diff) * 255.0));

}

void __global__ adjust_standard_deviation(float *strain_d, int width,
		int height, float std_deviation, float mean) {
	/**
	 * Get the x and y coordinates of the images
	 */
	int x = blockIdx.x;
	int y = threadIdx.x;
	float negative_threshold;
	float positive_threshold;

	if (POSITIVE_THRESHOLD_STD_DEVIATION_CUDA > 0) {
		positive_threshold = mean
				+ (float) POSITIVE_THRESHOLD_STD_DEVIATION_CUDA * std_deviation;
	} else {
		positive_threshold = POSITIVE_THRESHOLD_CONSTANT_CUDA;
	}

	if (NEGATIVE_THRESHOLD_STD_DEVIATION_CUDA > 0) {
		negative_threshold = mean
				- (float) NEGATIVE_THRESHOLD_STD_DEVIATION_CUDA * std_deviation;
	} else {
		negative_threshold = NEGATIVE_THRESHOLD_CONSTANT_CUDA;
	}

	/**
	 * Do the conversion b  = (a * min / diff) * 255;
	 */
	if (strain_d[y * width + x] <= negative_threshold) {
		strain_d[y * width + x] = STRAIN_VALUE_NEGATIVE_NOISE_CUDA;
	} else if (strain_d[y * width + x] >= positive_threshold) {
		strain_d[y * width + x] = STRAIN_VALUE_POSITIVE_NOISE_CUDA;
	}
}

int set_cuda_device(int device_id) {
	int device_count;
	cudaDeviceProp dprop;
	int selected_device;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	printf("Device count %d:\n", device_count);

	if (device_count == 0) {
		return 1;
	}
	if (device_id < 0 || device_id >= device_count) {
		selected_device = gpuGetMaxGflopsDeviceId();
	} else {
		selected_device = device_id;
	}
	/**
	 * if we have one or more devices then select second device for computation.
	 */
	cudaGetDeviceProperties(&dprop, selected_device);
	printf("Using Device number %d : %s\n", selected_device, dprop.name);

	cudaSetDevice(selected_device);
	return 0;
}

void initialize_ncc_slow(int window, float overlap, float displacement) {

}

void __global__ transpose_image(float *target, float *source, int width,
		int height) {
	int x = blockIdx.x;
	int y = threadIdx.x;

	target[x * height + y] = source[y * width + x];
}

void cuda_free_local(void *ptr, char *name) {
	char temp[80];
	cudaFreeHost(ptr);
	sprintf(temp, "Memory Free Error. %s %p", name, ptr);
	//checkCUDAError (temp);
}

void cuda_malloc_host(void **ptr, size_t ptr_size) {
	printf("cuda size %d\n", (int)ptr_size);
	checkCUDAError("Before allocation");
	cudaMallocHost(ptr, ptr_size);
	checkCUDAError("Memory allocation error.");
}

void cuda_copy_host(void *dst, void *src, size_t ptr_size) {
	cudaMemcpy(dst, src, ptr_size, cudaMemcpyHostToHost);
	checkCUDAError("cuda_copy_host error.");
}

void print_strain(FILE *fp, float* strain, int m, int n) {
	int i, j;

	for (i = 0; i < m; i++) {
		/*
		 printf ("[%d] ", i);
		 */
		for (j = 0; j < n; j++) {
			fprintf(fp, "%.4f ", *(strain + j + i * n));
		}
		fprintf(fp, "\n");
	}
}

void print_image(FILE *fp, short int* image, int m, int n) {
	int i, j;

	for (i = 0; i < m; i++) {
		/*
		 printf ("[%d] ", i);
		 */
		for (j = 0; j < n; j++) {
			fprintf(fp, "%d ", *(image + j * m + i));
		}
		fprintf(fp, "\n");
	}
}

void ncc_volume(int height, int width, short int * comp_matrix_short,
		short int *uncomp_matrix_short, float **cross_corr,
		float **displacement, unsigned char **out_strain, int *out_height,
		float *average_cross, float *average_strain, float *noise_percentage,
		float F0, float FS, int strain_or_displacement, int volume_size,
		int ncc_window, float ncc_overlap, float ncc_displacement) {
	float *t;
	float *corr;
	float strain_max;
	float strain_min;
	float displacement_max;
	float displacement_min;
	float *t_d;
	float *corr_d;
	short int *comp_matrix_d;
	short int *uncomp_matrix_d;
	float *EI_1_d;
	float *EI_1;
	float *EI_1_smoothed_d;
	float *strain;
	float *strain_d;
	float std_dev;
	unsigned char *strain_char;
	unsigned char *strain_char_d;
	float *strain_transpose;
	float *zhol;
	float *zhol_d;
	int NOF = 1;
	int diff_kernel;
	int comp_matrix_x;
	int comp_matrix_x_volume; //variable to store total RF lines across the complete volume size
	int comp_matrix_y;
	int uncomp_matrix_x;
	int uncomp_matrix_x_volume; //variable to store total RF lines across the complete volume size.
	int uncomp_matrix_y;
	float strain_average;
	float Win_T;
	int Win_S;
	float Over_T;
	int Over_S;
	float max_strain;
	int num_strain_points;
	int correlation_size;

	int comp_array_size;
	int uncomp_array_size;
	int op_array_size;
	int x;
	int *x_d;

	uncomp_matrix_x = comp_matrix_x = width;
	uncomp_matrix_y = comp_matrix_y = height;

	uncomp_matrix_x_volume = uncomp_matrix_x * volume_size;
	comp_matrix_x_volume = comp_matrix_x * volume_size;

	comp_array_size = sizeof(short int) * comp_matrix_x * comp_matrix_y
			* volume_size;
	uncomp_array_size = sizeof(short int) * uncomp_matrix_x * uncomp_matrix_y
			* volume_size;

	checkCUDAError("Allocation of memory 1");

	count_sampl_pts(uncomp_matrix_x, uncomp_matrix_y, ncc_window, ncc_overlap,
			ncc_displacement, &x, &Win_T, &Win_S, &Over_T, &Over_S, &max_strain,
			&num_strain_points, &correlation_size, F0, FS);

	//TODO allocate the new variable.

	allocate_constant(uncomp_matrix_x, uncomp_matrix_y, uncomp_matrix_y, Win_S,
			Over_S, max_strain, num_strain_points, volume_size);

	*out_height = x;

#ifdef PRINT_DEBUG
	printf("Input FO %.3f FS %.3f\n", F0, FS);
	printf(
			"Sample points for image of size x %d y %d ncc_window %d ncc_overlap %.2f ncc_displacement %.2f : %d\n",
			uncomp_matrix_x, uncomp_matrix_y, ncc_window, ncc_overlap,
			ncc_displacement, x);
#endif

	op_array_size = sizeof(float) * x * uncomp_matrix_x_volume;

	t = (float *) malloc(op_array_size);
	zhol = (float *) malloc(1000 * sizeof(float));
	//strain = (float *) malloc (op_array_size);
	diff_kernel = 8;

	checkCudaErrors(cudaMalloc((void **) &zhol_d, 1000 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &comp_matrix_d, comp_array_size));
	checkCudaErrors(cudaMalloc((void **) &uncomp_matrix_d, uncomp_array_size));
	checkCudaErrors(cudaMalloc((void **) &EI_1_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &EI_1_smoothed_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &corr_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &t_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &x_d, 1 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **) &strain_d, op_array_size));
	checkCudaErrors(
			cudaMalloc((void **) &strain_char_d,
					volume_size * x * uncomp_matrix_x * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void **) &strain_transpose, op_array_size));
	//cudaMallocHost((void**)&comp_matrix, comp_array_size); //freed in this function
	//cudaMallocHost((void**)&uncomp_matrix, comp_array_size); //freed in this function
	checkCudaErrors(cudaMallocHost((void**) &strain, op_array_size)); //freed in this function
	checkCudaErrors(cudaMallocHost((void**) &EI_1, op_array_size)); //freed in caller
	checkCudaErrors(
			cudaMallocHost((void**) &strain_char,
					volume_size * x * uncomp_matrix_x * sizeof(unsigned char))); //freed in caller
	checkCudaErrors(cudaMallocHost((void**) &corr, op_array_size)); //freed in caller
	checkCUDAError("Allocation of memory");

	//printf ("Address after allocation %p\n", strain_char);

	//for (i = 0; i < width; i++) {
	//	for (j = 0; j < height; j++) {
	//	    uncomp_matrix [ i * height + j ] = uncomp_matrix_short [ i * height + j ];
	//		comp_matrix [i * height + j] = comp_matrix_short [i * height + j];
	//	}	
	//}

	checkCudaErrors(cudaMemset(EI_1_d, 0, op_array_size));

	checkCudaErrors(
			cudaMemcpy(comp_matrix_d, comp_matrix_short, comp_array_size,
					cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMemcpy(uncomp_matrix_d, uncomp_matrix_short, uncomp_array_size,
					cudaMemcpyHostToDevice));

	/**
	 * Perform normalized cross correlation
	 */

	normalizecrosscorr_jumbo<<<uncomp_matrix_x_volume, x, 0>>>(comp_matrix_d,
			uncomp_matrix_d, t_d, corr_d, EI_1_d);

	checkCUDAError("Calling normalized cross corr");

	median_filter<<<uncomp_matrix_x_volume, x, 0>>>(EI_1_d, EI_1_smoothed_d,
			MEDIAN_FILT_WIDTH, MEDIAN_FILT_HEIGHT, comp_matrix_x, x,
			comp_matrix_x / NOF, x, zhol_d);

	checkCUDAError("Calling median filter");

	moving_average<<<uncomp_matrix_x_volume, x, 0>>>(EI_1_smoothed_d, EI_1_d,
			MOVING_AVERAGE_DEPTH, uncomp_matrix_x);

	checkCUDAError("Calling moving average");

	checkCudaErrors(
			cudaMemcpy(EI_1, EI_1_d, op_array_size, cudaMemcpyDeviceToHost));

	*displacement = EI_1;

	st_clear<<<uncomp_matrix_x_volume, x, 0>>>(strain_d, uncomp_matrix_x);

	checkCUDAError("Calling st_clear");

	st_LSQSE<<<comp_matrix_x_volume, x - diff_kernel, 0>>>(EI_1_d, strain_d,
			diff_kernel, uncomp_matrix_x);

	checkCUDAError("Calling st_LSQSE");

	/**
	 * copy the strain back to the main memory
	 */

	checkCudaErrors(
			cudaMemcpy(strain, strain_d, op_array_size,
					cudaMemcpyDeviceToHost));

	/**
	 * Find minimum and maximum
	 */

	find_min_max_dev(strain, uncomp_matrix_x_volume, x, &strain_min,
			&strain_max, &std_dev, &strain_average, noise_percentage);

#ifdef PRINT_DEBUG
	printf(
			"Min and Max for the strain image are min %.3f max %.3f std_deviation %.3f\n",
			strain_min, strain_max, std_dev);
#endif

	*average_strain = strain_average;
#ifdef PRINT_DEBUG
	printf("Standard deviation %.3f Strain average %.3f\n", std_dev,
			strain_average);
#endif

	if (strain_average < 0) {

		printf("Average is greater than 0 so recomputing\n");

		normalizecrosscorr_jumbo<<<uncomp_matrix_x_volume, x, 0>>>(
				uncomp_matrix_d, comp_matrix_d, t_d, corr_d, EI_1_d);

		checkCUDAError("Calling normalized cross corr 2");

		median_filter<<<uncomp_matrix_x_volume, x, 0>>>(EI_1_d, EI_1_smoothed_d,
				MEDIAN_FILT_WIDTH, MEDIAN_FILT_HEIGHT, comp_matrix_x, x,
				comp_matrix_x / NOF, x, zhol_d);

		checkCUDAError("Calling median filter 2");

		moving_average<<<uncomp_matrix_x_volume, x, 0>>>(EI_1_smoothed_d,
				EI_1_d, MOVING_AVERAGE_DEPTH, uncomp_matrix_x);

		checkCUDAError("Calling moving average 2");

		cudaMemcpy(EI_1, EI_1_d, op_array_size, cudaMemcpyDeviceToHost);

		*displacement = EI_1;

		//No need to calculate strain again
		if (strain_or_displacement == 0) {
			st_clear<<<uncomp_matrix_x_volume, x, 0>>>(strain_d,
					uncomp_matrix_x);

			checkCUDAError("Calling st_clear 2");

			st_LSQSE<<<comp_matrix_x_volume, x - diff_kernel, 0>>>(EI_1_d,
					strain_d, diff_kernel, uncomp_matrix_x);

			checkCUDAError("Calling st_LSQSE 2");

			/**        
			 * This transfer back is attached to the processing of the device
			 */

			/**
			 * copy the strain back to the main memory
			 */

			cudaMemcpy(strain, strain_d, op_array_size, cudaMemcpyDeviceToHost);

			/**
			 * Find minimum and maximum
			 */

			find_min_max_dev(strain, uncomp_matrix_x_volume, x, &strain_min,
					&strain_max, &std_dev, &strain_average, noise_percentage);
#ifdef PRINT_DEBUG
			printf(
					"Min and Max for the strain image are min %.3f max %.3f std_deviation %.3f\n",
					strain_min, strain_max, std_dev);
#endif
		}
	}

#ifdef PRINT_DEBUG
	printf("Standard deviation %.3f Strain average %.3f\n", std_dev,
			strain_average);
#endif

//TODO adjust standard deviation has to be changed to accomodate volume of data.

	/**
	 * Perform low and high pass filtering
	 */
	adjust_standard_deviation<<<uncomp_matrix_x_volume, x, 0>>>(strain_d,
			uncomp_matrix_x_volume, x, std_dev, strain_average);

	if (POSITIVE_THRESHOLD_STD_DEVIATION > 0) {
		strain_max = strain_average
				+ (float) POSITIVE_THRESHOLD_STD_DEVIATION * std_dev;
	} else {
		strain_max = POSITIVE_THRESHOLD_CONSTANT;
	}

	if (NEGATIVE_THRESHOLD_STD_DEVIATION > 0) {
		strain_min = strain_average
				- (float) NEGATIVE_THRESHOLD_STD_DEVIATION * std_dev;
	} else {
		strain_min = NEGATIVE_THRESHOLD_CONSTANT;
	}
	//

	/**
	 * Map it to 0 - 255 scale
	 */
//TODO colormap has to be changed to accomodate volume of data.
	if ((strain_max - strain_min) != 0) {

		if (strain_or_displacement == 0) {
			map_255_colormap<<<uncomp_matrix_x_volume, x, 0>>>(strain_d,
					strain_char_d, uncomp_matrix_x_volume, x, strain_min,
					strain_max - strain_min);
		} else {
			find_min_max_displacement(EI_1, uncomp_matrix_x_volume, x,
					&displacement_min, &displacement_max);
			//We are using strain_char_d to save memory
			//TODO name should be changed to avoid confusions in the future.
			map_255_colormap<<<uncomp_matrix_x_volume, x, 0>>>(EI_1_d,
					strain_char_d, uncomp_matrix_x_volume, x, displacement_min,
					displacement_max - displacement_min);
		}
		checkCUDAError("Calling map_255_colormap");
	} else {
		printf("Error in calculating the image strain_max %f strain_min %f\n",
				strain_max, strain_min);
	}

	cudaMemcpy(t, t_d, op_array_size, cudaMemcpyDeviceToHost);

	cudaMemcpy(corr, corr_d, op_array_size, cudaMemcpyDeviceToHost);

	//strain_char_d will contain displacement if strain_or_displacement != 0
	cudaMemcpy(strain_char, strain_char_d,
			x * uncomp_matrix_x * sizeof(char) * volume_size,
			cudaMemcpyDeviceToHost);

	//cudaMemcpy (strain, strain_d, op_array_size, cudaMemcpyDeviceToHost);

	checkCUDAError("cudamemcpy from host to device");

	*out_strain = strain_char;

	average_cross_correlation(corr, uncomp_matrix_x_volume, x, average_cross);

#ifdef PRINT_DEBUG_OUTPUT_NCC
	char file_name[50];
	sprintf (file_name, "C:\\ei_out\\strain_%d.txt", global_file_count);
	FILE *fp = fopen (file_name, "w");
	if (fp == NULL) {
		printf ("Error\n");
		exit(1);
	}
	print_strain (fp, strain, x, comp_matrix_x);
//      print_disparity (fp, disparity_1_local + i * m * n, m, n);
	fclose (fp);

	sprintf (file_name, "C:\\ei_out\\uncompress_1_%d.txt", global_file_count);
	fp = fopen (file_name, "w");
	if (fp == NULL) {
		printf ("Error\n");
		exit(1);
	}

	print_image (fp, uncomp_matrix_short, comp_matrix_y, comp_matrix_x);
	fclose(fp);

	sprintf (file_name, "C:\\ei_out\\compress_1_%d.txt", global_file_count);
	global_file_count++;
	fp = fopen (file_name, "w");
	if (fp == NULL) {
		printf ("Error\n");
		exit(1);
	}
	print_image (fp, comp_matrix_short, comp_matrix_y, comp_matrix_x);
	fclose(fp);

	if (global_file_count == 40) {
		exit (11);
	}
#endif

	free(t);
//removed this free
///    free (corr);
	free(zhol);

	cudaFree(zhol_d);
	cudaFree(comp_matrix_d);
	cudaFree(uncomp_matrix_d);
	cudaFree(EI_1_d);
	cudaFree(EI_1_smoothed_d);
	cudaFree(corr_d);
	cudaFree(t_d);
	cudaFree(x_d);
	cudaFree(strain_d);
	cudaFree(strain_char_d);
	cudaFree(strain_transpose);
	//Free it in the main function
	*cross_corr = corr;
	//cudaFreeHost (comp_matrix);
	cudaFreeHost(strain);
	//cudaFreeHost (uncomp_matrix);
	checkCUDAError("Error before exit");
}

void ncc_slow(int height, int width, short int * comp_matrix_short,
		short int *uncomp_matrix_short, float **cross_corr,
		float **displacement, unsigned char **out_strain, int *out_height,
		float *average_cross, float *average_strain, float *noise_percentage,
		float F0, float FS, int strain_or_displacement, int NOF, int ncc_window,
		float ncc_overlap, float ncc_displacement) {
	cudaStream_t stream_ncc;
	float *t;
	float *corr;
	float strain_max;
	float strain_min;
	float displacement_max;
	float displacement_min;
	float *t_d;
	float *corr_d;
	short int *comp_matrix_d;
	short int *uncomp_matrix_d;
	//float *comp_matrix;
	//float *uncomp_matrix;
	float *EI_1_d;
	float *EI_1;
	float *EI_1_smoothed_d;
	float *strain;
	float *strain_d;
	float std_dev;
	unsigned char *strain_char;
	unsigned char *strain_char_d;
	float *strain_transpose;
	float *zhol;
	float *zhol_d;
	//int NOF = 1;
	int diff_kernel;
	//int i;
	int comp_matrix_x;
	int comp_matrix_y;
	int uncomp_matrix_x;
	int uncomp_matrix_y;
	float strain_average;
	float Win_T;
	int Win_S;
	float Over_T;
	int Over_S;
	float max_strain;
	int num_strain_points;
	int correlation_size;
	int k;
	int strain_size_float;
	int strain_size_char;
	short int *comp_matrix_short_h;
	short int *uncomp_matrix_short_h;
	float *EI_1_return;
	float *corr_return;
	unsigned char *strain_char_return;

	int comp_array_size;
	int uncomp_array_size;
	int op_array_size;
	int x;
	int *x_d;

#ifdef CUDA_EVENT_TIMER
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
#endif

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(start, 0);
#endif

	uncomp_matrix_x = comp_matrix_x = width;
	uncomp_matrix_y = comp_matrix_y = height;

	//printf ("uncomp_matrix_x %d uncomp_matrix_y %d NOF %d\n", uncomp_matrix_x, uncomp_matrix_y, NOF);

	comp_array_size = sizeof(short int) * comp_matrix_x * comp_matrix_y;
	uncomp_array_size = sizeof(short int) * uncomp_matrix_x * uncomp_matrix_y;

	count_sampl_pts(uncomp_matrix_x, uncomp_matrix_y, ncc_window, ncc_overlap,
			ncc_displacement, &x, &Win_T, &Win_S, &Over_T, &Over_S, &max_strain,
			&num_strain_points, &correlation_size, F0, FS);

	allocate_constant(uncomp_matrix_x / NOF, uncomp_matrix_y, uncomp_matrix_y,
			Win_S, Over_S, max_strain, num_strain_points, 1);
	checkCUDAError("Allocation of memory");
	*out_height = x;

#ifdef PRINT_DEBUG
	printf("Input FO %.3f FS %.3f\n", F0, FS);
	printf(
			"Sample points for image of size x %d y %d ncc_window %d ncc_overlap %.2f ncc_displacement %.2f : %d\n",
			uncomp_matrix_x, uncomp_matrix_y, ncc_window, ncc_overlap,
			ncc_displacement, x);
#endif
	op_array_size = sizeof(float) * x * uncomp_matrix_x;

	//t = (float *) malloc (op_array_size);    
	zhol = (float *) malloc(1000 * sizeof(float));
	//strain = (float *) malloc (op_array_size);
	//Create a stream
	checkCudaErrors(cudaStreamCreate(&stream_ncc));
	diff_kernel = 8;

	checkCudaErrors(cudaMalloc((void **) &zhol_d, 1000 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &comp_matrix_d, comp_array_size));
	checkCudaErrors(cudaMalloc((void **) &uncomp_matrix_d, uncomp_array_size));
	checkCudaErrors(cudaMalloc((void **) &EI_1_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &EI_1_smoothed_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &corr_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &t_d, op_array_size));
	checkCudaErrors(cudaMalloc((void **) &x_d, 1 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **) &strain_d, op_array_size));
	checkCudaErrors(
			cudaMalloc((void **) &strain_char_d,
					x * uncomp_matrix_x * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void **) &strain_transpose, op_array_size));
	checkCudaErrors(
			cudaMallocHost((void **) &comp_matrix_short_h, comp_array_size));
	checkCudaErrors(
			cudaMallocHost((void **) &uncomp_matrix_short_h, comp_array_size));
	//cudaMallocHost((void**)&comp_matrix, comp_array_size); //freed in this function
	//cudaMallocHost((void**)&uncomp_matrix, comp_array_size); //freed in this function
	checkCudaErrors(cudaMallocHost((void**) &strain, op_array_size)); //freed in this function
	checkCudaErrors(cudaMallocHost((void**) &EI_1, op_array_size)); //freed in caller
	checkCudaErrors(
			cudaMallocHost((void**) &strain_char,
					x * uncomp_matrix_x * sizeof(unsigned char))); //freed in caller
	checkCudaErrors(cudaMallocHost((void**) &corr, op_array_size)); //freed in caller
	checkCudaErrors(cudaMallocHost((void **) &t, op_array_size));
	checkCUDAError("Allocation of memory");

	//printf ("Address after allocation %p\n", strain_char);

	//for (i = 0; i < width; i++) {
	//	for (j = 0; j < height; j++) {
	//	    uncomp_matrix [ i * height + j ] = uncomp_matrix_short [ i * height + j ];
	//		comp_matrix [i * height + j] = comp_matrix_short [i * height + j];
	//	}	
	//}

	memcpy(comp_matrix_short_h, comp_matrix_short, comp_array_size);
	memcpy(uncomp_matrix_short_h, uncomp_matrix_short, comp_array_size);
	checkCudaErrors(cudaMemset(EI_1_d, 0, op_array_size));
	checkCudaErrors(
			cudaMemcpyAsync(comp_matrix_d, comp_matrix_short_h, comp_array_size,
					cudaMemcpyHostToDevice, stream_ncc));
	checkCudaErrors(
			cudaMemcpyAsync(uncomp_matrix_d, uncomp_matrix_short_h,
					uncomp_array_size, cudaMemcpyHostToDevice, stream_ncc));
#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for initial allocation: %f ms\n", time);
#endif

	cudaThreadSynchronize();

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(start, 0);
#endif

	/**
	 * Perform normalized cross correlation
	 */
	for (k = 0; k < NOF; k++) {
		normalizecrosscorr_jumbo<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
				comp_matrix_d + k * (comp_matrix_x / NOF) * comp_matrix_y,
				uncomp_matrix_d + k * (comp_matrix_x / NOF) * comp_matrix_y,
				t_d + k * (comp_matrix_x / NOF) * x,
				corr_d + k * (comp_matrix_x / NOF) * x,
				EI_1_d + k * (comp_matrix_x / NOF) * x);
		checkCUDAError("Calling normalized cross corr");
	}
#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for calculating NCC: %f ms\n", time);
#endif

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(start, 0);
#endif
	for (k = 0; k < NOF; k++) {
		median_filter<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
				EI_1_d + k * (comp_matrix_x / NOF) * x,
				EI_1_smoothed_d + k * (comp_matrix_x / NOF) * x,
				MEDIAN_FILT_WIDTH, MEDIAN_FILT_HEIGHT, comp_matrix_x / NOF, x,
				comp_matrix_x / NOF, x, zhol_d);
		checkCUDAError("Calling median filter");
	}
#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for calculating median filter: %f ms\n", time);
#endif
#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(start, 0);
#endif
	for (k = 0; k < NOF; k++) {
		moving_average<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
				EI_1_smoothed_d + k * (comp_matrix_x / NOF) * x,
				EI_1_d + k * (comp_matrix_x / NOF) * x, MOVING_AVERAGE_DEPTH,
				uncomp_matrix_x / NOF);

		checkCUDAError("Calling moving average");
	}

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for calculating moving average: %f ms\n", time);
#endif
	checkCudaErrors(
			cudaMemcpyAsync(EI_1, EI_1_d, op_array_size, cudaMemcpyDeviceToHost,
					stream_ncc));

	EI_1_return = (float *) malloc(op_array_size);

//This temporary memory is needed so that all cuda host allocated memory is limited to this function and we don't have to deal with freeing
//the memory somewhere else.

	if (EI_1_return == NULL) {
		printf("Error allocating EI_1_return\n");
		exit(1);
	}

	cudaStreamSynchronize(stream_ncc);
	memcpy(EI_1_return, EI_1, op_array_size);

	*displacement = EI_1_return;

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(start, 0);
#endif	

	st_clear<<<uncomp_matrix_x, x, 0, stream_ncc>>>(strain_d, uncomp_matrix_x);

	checkCUDAError("Calling st_clear");

	assert(x - diff_kernel > 0);

	for (k = 0; k < NOF; k++) {
#ifdef PRINT_DEBUG
		printf("%d, %d\n", comp_matrix_x / NOF, x - diff_kernel);
#endif
		st_LSQSE<<<comp_matrix_x / NOF, x - diff_kernel, 0, stream_ncc>>>(
				EI_1_d + k * (comp_matrix_x / NOF) * x,
				strain_d + k * (comp_matrix_x / NOF) * x, diff_kernel,
				uncomp_matrix_x / NOF);
		checkCUDAError("Calling st_LSQSE");
	}

	checkCUDAError("Calling st_LSQSE");
	/**
	 * copy the strain back to the main memory
	 */
#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for calculating strain: %f ms\n", time);
#endif

	checkCudaErrors(
			cudaMemcpyAsync(strain, strain_d, op_array_size,
					cudaMemcpyDeviceToHost, stream_ncc));

	/**
	 * Find minimum and maximum
	 */
	cudaStreamSynchronize(stream_ncc);

	find_min_max_dev(strain, uncomp_matrix_x, x, &strain_min, &strain_max,
			&std_dev, &strain_average, noise_percentage);

#ifdef PRINT_DEBUG
	printf(
			"Min and Max for the strain image are min %.3f max %.3f std_deviation %.3f\n",
			strain_min, strain_max, std_dev);
#endif

	*average_strain = strain_average;

#ifdef PRINT_DEBUG
	printf("Standard deviation %.3f Strain average %.3f\n", std_dev,
			strain_average);
#endif

	if (strain_average < 0) {
		printf("test\n");
		printf("Average is less than 0 so recomputing\n");

		for (k = 0; k < NOF; k++) {
			normalizecrosscorr_jumbo<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
					uncomp_matrix_d
							+ k * (uncomp_matrix_x / NOF) * uncomp_matrix_y,
					comp_matrix_d + k * (comp_matrix_x / NOF) * comp_matrix_y,
					t_d + k * (comp_matrix_x / NOF) * x,
					corr_d + k * (comp_matrix_x / NOF) * x,
					EI_1_d + k * (comp_matrix_x / NOF) * x);
			checkCUDAError("Calling normalized cross corr");
		}

		/*normalizecrosscorr_jumbo <<<uncomp_matrix_x, x, 0>>> 
		 (uncomp_matrix_d, comp_matrix_d, t_d, corr_d, EI_1_d);

		 checkCUDAError ("Calling normalized cross corr 2");*/

		for (k = 0; k < NOF; k++) {
			median_filter<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
					EI_1_d + k * (comp_matrix_x / NOF) * x,
					EI_1_smoothed_d + k * (comp_matrix_x / NOF) * x,
					MEDIAN_FILT_WIDTH, MEDIAN_FILT_HEIGHT, comp_matrix_x / NOF,
					x, comp_matrix_x / NOF, x, zhol_d);
			checkCUDAError("Calling median filter 2");
		}
		printf("test\n");
		/*median_filter<<<uncomp_matrix_x, x, 0>>>
		 (EI_1_d, EI_1_smoothed_d, MEDIAN_FILT_WIDTH, MEDIAN_FILT_HEIGHT, comp_matrix_x, x,
		 comp_matrix_x/NOF, x, zhol_d);

		 checkCUDAError ("Calling median filter 2");*/

		for (k = 0; k < NOF; k++) {
			moving_average<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
					EI_1_smoothed_d + k * (comp_matrix_x / NOF) * x,
					EI_1_d + k * (comp_matrix_x / NOF) * x,
					MOVING_AVERAGE_DEPTH, uncomp_matrix_x / NOF);

			checkCUDAError("Calling moving average 2");
		}

		/*moving_average<<<uncomp_matrix_x, x, 0>>>
		 (EI_1_smoothed_d, EI_1_d, MOVING_AVERAGE_DEPTH, uncomp_matrix_x);

		 checkCUDAError ("Calling moving average 2");*/

		cudaMemcpyAsync(EI_1, EI_1_d, op_array_size, cudaMemcpyDeviceToHost,
				stream_ncc);
		cudaStreamSynchronize(stream_ncc);

		//EI_1_return is already allocated.
		memcpy(EI_1_return, EI_1, op_array_size);

		*displacement = EI_1_return;

		//No need to calculate strain again
		if (strain_or_displacement == 0) {
			st_clear<<<uncomp_matrix_x, x, 0, stream_ncc>>>(strain_d,
					uncomp_matrix_x);

			checkCUDAError("Calling st_clear 2");

			for (k = 0; k < NOF; k++) {

				st_LSQSE<<<comp_matrix_x / NOF, x - diff_kernel, 0, stream_ncc>>>(
						EI_1_d + k * (comp_matrix_x / NOF) * x,
						strain_d + k * (comp_matrix_x / NOF) * x, diff_kernel,
						uncomp_matrix_x / NOF);

			}

			checkCUDAError("Calling st_LSQSE 2");

			/**        
			 * This transfer back is attached to the processing of the device
			 */

			/**
			 * copy the strain back to the main memory
			 */

			cudaMemcpyAsync(strain, strain_d, op_array_size,
					cudaMemcpyDeviceToHost, stream_ncc);
			cudaStreamSynchronize(stream_ncc);

			/**
			 * Find minimum and maximum
			 */

			find_min_max_dev(strain, uncomp_matrix_x, x, &strain_min,
					&strain_max, &std_dev, &strain_average, noise_percentage);

#ifdef PRINT_DEBUG
			printf(
					"Min and Max for the strain image are min %.3f max %.3f std_deviation %.3f\n",
					strain_min, strain_max, std_dev);
#endif
		}
	}

#ifdef PRINT_DEBUG
	printf("Standard deviation %.3f Strain average %.3f\n", std_dev,
			strain_average);
#endif

	strain_size_float = uncomp_matrix_x / NOF * x;
	strain_size_char = uncomp_matrix_x / NOF * x;

//For loop will do conversion independently for each strain values separately.
	for (k = 0; k < NOF; k++) {

		find_min_max_dev(strain + strain_size_float * k, uncomp_matrix_x / NOF,
				x, &strain_min, &strain_max, &std_dev, &strain_average,
				noise_percentage);

#ifdef PRINT_DEBUG	
		printf(
				"Min and Max for the strain image are min %.3f max %.3f std_deviation %.3f Average %.3f\n",
				strain_min, strain_max, std_dev, strain_average);
#endif
		/**
		 * Perform low and high pass filtering
		 */
#ifdef CUDA_EVENT_TIMER
		cudaEventRecord(start, 0);
#endif	

		adjust_standard_deviation<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
				strain_d + strain_size_float * k, uncomp_matrix_x / NOF, x,
				std_dev, strain_average);

#ifdef CUDA_EVENT_TIMER
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf ("Time for adjusting standard deviation: %f ms\n", time);
#endif

		if (POSITIVE_THRESHOLD_STD_DEVIATION > 0) {
			strain_max = strain_average
					+ (float) POSITIVE_THRESHOLD_STD_DEVIATION * std_dev;
		} else {
			strain_max = POSITIVE_THRESHOLD_CONSTANT;
		}

		if (NEGATIVE_THRESHOLD_STD_DEVIATION > 0) {
			strain_min = strain_average
					- (float) NEGATIVE_THRESHOLD_STD_DEVIATION * std_dev;
		} else {
			strain_min = NEGATIVE_THRESHOLD_CONSTANT;
		}
		//

		/**
		 * Map it to 0 - 255 scale
		 */
#ifdef CUDA_EVENT_TIMER
		cudaEventRecord(start, 0);
#endif		 
		if ((strain_max - strain_min) != 0) {

			if (strain_or_displacement == 0) {
				map_255_colormap<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
						strain_d + strain_size_float * k,
						strain_char_d + strain_size_char * k,
						uncomp_matrix_x / NOF, x, strain_min,
						strain_max - strain_min);
			} else {
				find_min_max_displacement(EI_1 + strain_size_float * k,
						uncomp_matrix_x / NOF, x, &displacement_min,
						&displacement_max);
				//We are using strain_char_d to save memory
				//TODO name should be changed to avoid confusions in the future.
				map_255_colormap<<<uncomp_matrix_x / NOF, x, 0, stream_ncc>>>(
						EI_1_d + strain_size_float * k,
						strain_char_d + strain_size_char * k,
						uncomp_matrix_x / NOF, x, displacement_min,
						displacement_max - displacement_min);
			}
			checkCUDAError("Calling map_255_colormap");
		} else {
			printf(
					"Error in calculating the image k %d strain_max %f strain_min %f\n",
					k, strain_max, strain_min);
		}
	}

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for adjusting colormaps: %f ms\n", time);
#endif

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(start, 0);
#endif

	cudaMemcpyAsync(t, t_d, op_array_size, cudaMemcpyDeviceToHost, stream_ncc);

	cudaMemcpyAsync(corr, corr_d, op_array_size, cudaMemcpyDeviceToHost,
			stream_ncc);

	//strain_char_d will contain displacement if strain_or_displacement != 0
	cudaMemcpyAsync(strain_char, strain_char_d,
			x * uncomp_matrix_x * sizeof(char), cudaMemcpyDeviceToHost,
			stream_ncc);

	//cudaMemcpy (strain, strain_d, op_array_size, cudaMemcpyDeviceToHost);

	cudaStreamSynchronize(stream_ncc);

	checkCUDAError("cudamemcpy from host to device");

//This temporary memory is needed so that all cuda host allocated memory is limited to this function and we don't have to deal with freeing
//the memory somewhere else.

	strain_char_return = (unsigned char *) malloc(
			x * uncomp_matrix_x * sizeof(char));

	if (strain_char_return == NULL) {
		printf("Error allocating memory for strain_char_return\n");
		exit(1);
	}

	memcpy(strain_char_return, strain_char, x * uncomp_matrix_x * sizeof(char));

	*out_strain = strain_char_return;

#ifdef CUDA_EVENT_TIMER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for Final copy: %f ms\n", time);
#endif

	average_cross_correlation(corr, uncomp_matrix_x, x, average_cross);

#ifdef PRINT_DEBUG_OUTPUT_NCC
	char file_name[50];
	sprintf (file_name, "C:\\ei_out\\strain_%d.txt", global_file_count);
	FILE *fp = fopen (file_name, "w");
	if (fp == NULL) {
		printf ("Error\n");
		exit(1);
	}
	print_strain (fp, strain, x, comp_matrix_x);
//      print_disparity (fp, disparity_1_local + i * m * n, m, n);
	fclose (fp);

	sprintf (file_name, "C:\\ei_out\\uncompress_1_%d.txt", global_file_count);
	fp = fopen (file_name, "w");
	if (fp == NULL) {
		printf ("Error\n");
		exit(1);
	}

	print_image (fp, uncomp_matrix_short, comp_matrix_y, comp_matrix_x);
	fclose(fp);

	sprintf (file_name, "C:\\ei_out\\compress_1_%d.txt", global_file_count);
	global_file_count++;
	fp = fopen (file_name, "w");
	if (fp == NULL) {
		printf ("Error\n");
		exit(1);
	}
	print_image (fp, comp_matrix_short, comp_matrix_y, comp_matrix_x);
	fclose(fp);

	if (global_file_count == 40) {
		exit (11);
	}
#endif

//removed this free
///    free (corr);
	free(zhol);

	cudaFree(zhol_d);
	cudaFree(comp_matrix_d);
	cudaFree(uncomp_matrix_d);
	cudaFree(EI_1_d);
	cudaFree(EI_1_smoothed_d);
	cudaFree(corr_d);
	cudaFree(t_d);
	cudaFree(x_d);
	cudaFree(strain_d);
	cudaFree(strain_char_d);
	cudaFree(strain_transpose);
	//Destroy the stream
	checkCudaErrors(cudaStreamDestroy(stream_ncc));
	//Free it in the main function

	//TODO change this so that cudamemhost is freed somewhere.

	corr_return = (float *) malloc(op_array_size);
	if (corr_return == NULL) {
		printf("correlation return value could not be allocated memory\n");
		exit(1);
	}

	memcpy(corr_return, corr, op_array_size);
	*cross_corr = corr_return;

	//cudaFreeHost (comp_matrix);
	cudaFreeHost(strain);
	cudaFreeHost(t);
	cudaFreeHost(comp_matrix_short_h);
	cudaFreeHost(uncomp_matrix_short_h);
	cudaFreeHost(EI_1);
	cudaFreeHost(corr);
	cudaFreeHost(strain_char);
	//cudaFreeHost (uncomp_matrix);
	checkCUDAError("Error before exit");
}

#ifndef round
/*__device__ double round(double round);
 __device__ double round(double value)
 {
 if (value < 0)
 return -(floor(-value + 0.5));
 else
 return   floor( value + 0.5);
 }*/

#endif

__device__ float norm_mean(short int* input, int size) {
	int i;
	float sum = 0;
	for (i = 0; i < size; i++) {
		sum += input[i];
	}
	return ((float) sum / size);
}

__device__ float norm_mean_2(short int* input, int start, int size) {
	int i;
	float sum = 0;
	for (i = start; i < (start + size); i++) {
		sum += input[i];
	}
	return ((float) sum / size);
}

__device__ void normalizecrosscorr_2(short int *comp, short int *uncomp,
		int comp_x, int uncomp_x, float *gamma, float f_mean, float t_mean) {
	int loop_count;
	int inner_loop;
	//float f_mean;
	//float t_mean;
	float normalize_comp;
	float normalize_uncomp;
	float numerator;
	float denominator_1;
	float denominator_2;
	float denominator;
	//f_mean = norm_mean(uncomp, uncomp_x);
	//t_mean = norm_mean(comp, comp_x);
	numerator = 0;
	denominator_1 = 0;
	denominator_2 = 0;

	for (loop_count = 0; loop_count < (comp_x - uncomp_x + 1); loop_count++) {
		denominator_1 = 0;
		denominator_2 = 0;
		numerator = 0;
		t_mean = norm_mean_2(comp, loop_count, uncomp_x);
		for (inner_loop = 0; inner_loop < uncomp_x; inner_loop++) {
			normalize_uncomp = (uncomp[inner_loop] - f_mean);
			normalize_comp = (comp[inner_loop + loop_count] - t_mean);
			denominator_1 += normalize_uncomp * normalize_uncomp;
			denominator_2 += normalize_comp * normalize_comp;
			numerator += normalize_uncomp * normalize_comp;
		}
		denominator = sqrt((denominator_1 * denominator_2));
		gamma[loop_count] = numerator / denominator;
	}
}

__device__ int max_index(float *a, int size) {
	int count = 1;
	int index = 0;
	float temp;
	if (size > 0) {
		temp = a[0];
		for (; count < (size); count++) {
			if (temp < a[count]) {
				temp = a[count];
				index = count;
			}
		}
		return index;
	}
	return -1;
}
__device__ float max_value(float *a, int size) {
	int count = 1;
	float temp;
	if (size > 0) {
		temp = a[0];
		for (; count < (size); count++) {
			if (temp < a[count]) {
				temp = a[count];
			}
		}
	}
	return temp;
}

void count_sampl_pts(int uncomp_matrix_x, int uncomp_matrix_y, int window,
		float overlap, float displacement, int *x, float *Win_T, int *Win_S,
		float *Over_T, int *Over_S, float *max_strain, int *num_strain_points,
		int *correlation_size, float F0, float FS) {
	float f0 = F0;
	float fs = FS;
	int range_start;
	int range_end;
	int search_start;
	int search_end;

	*Win_T = window / f0;
	*Win_S = (int) ceil((*Win_T) * fs);
	*Over_T = overlap / f0;
	*Over_S = (int) ceil((*Over_T) * fs);
	int c = 1540000;
	*max_strain = (float) (ceil(((displacement * 2) / c) * fs * 1e6));

	float num_samples = (float) uncomp_matrix_y;
	//num_lines = uncomp_matrix_x;
	*num_strain_points = (int) (floorf((num_samples - (*Win_S)) / (*Over_S)));

	int starting_point = (int) ceil(((*max_strain) - 2) / (2 * (*Over_S)) + 1);

	range_start = (5) * (*Over_S);
	range_end = (*Win_S) + (5) * (*Over_S) - 1;
	search_start = (int) round(((5) * (*Over_S) + 1 - (*max_strain) / 2)) - 1;
	search_end = (int) round((*Win_S) + (5) * (*Over_S) + (*max_strain) / 4)
			- 1;

	*correlation_size = (search_end - search_start) - (range_end - range_start)
			+ 1;

	*x = ((*num_strain_points) - 5);
	/*printf ("f0 = %.3f\n", f0);
	 printf ("fs = %.3f\n", fs);
	 printf ("Win_T = %.3f\n", *Win_T);
	 printf ("Win_S = %d\n", *Win_S);
	 printf ("Over_T = %.3f\n", *Over_T);
	 printf ("Over_S = %d\n", *Over_S);
	 printf ("max_strain = %.3f\n", *max_strain);
	 printf ("starting_point = %d\n", starting_point);
	 printf ("num_strain_points = %d\n", *num_strain_points);
	 printf ("correlation_size = %d\n", *correlation_size);
	 fflush(stdout);*/
}

// I added EI_1 to the outputs
/*
 * pt_index is row identity
 * line_index is column identity
 */
__global__ void normalizecrosscorr_jumbo(short int *comp_matrix,
		short int *uncomp_matrix, float *t, float *corr, float *EI_1) {
	float corrar[100];
	int corrar_max_index;
	float y0;
	float y1;
	float y2;
	float w0;
	float w0_ratio;
	float theta;
	float delta2;
	float f_mean;
	float t_mean;
	int line_index = blockIdx.x;
	int pt_index = threadIdx.x;
	int range_start;
	int range_end;
	int search_start;
	int search_end;

	if (pt_index < 4) {
		*(EI_1 + uncomp_matrix_x_const * pt_index * volume_size_const
				+ line_index) = -1 - max_strain_const / 2;
		*(t + uncomp_matrix_x_const * pt_index * volume_size_const + line_index) =
				0;
		*(corr + uncomp_matrix_x_const * pt_index * volume_size_const
				+ line_index) = 0;
	} else {
		if (pt_index < (num_strain_points_const - 5)) {

			range_start = (pt_index) * Over_S_const;

			range_end = Win_S_const + (pt_index) * Over_S_const - 1;
			if (range_end >= (uncomp_matrix_y_const)) {
				range_end = (uncomp_matrix_y_const - 1);
			}

			search_start = round(
					((pt_index) * Over_S_const + 1 - max_strain_const / 2)) - 1;

			if (search_start < 4) {
				search_start = 4;
			}
			search_end = round(
					Win_S_const + (pt_index) * Over_S_const
							+ max_strain_const / 2) - 1;

			if (search_end >= (uncomp_matrix_y_const)) {
				search_end = (uncomp_matrix_y_const - 1);
			}

			f_mean = norm_mean(
					(uncomp_matrix + N_const * line_index + range_start),
					((range_end - range_start) + 1));
			t_mean = norm_mean(
					(comp_matrix + N_const * line_index + search_start),
					((search_end - search_start) + 1));

			normalizecrosscorr_2(
					(comp_matrix + N_const * line_index + search_start),
					(uncomp_matrix + N_const * line_index + range_start),
					((search_end - search_start) + 1),
					((range_end - range_start) + 1), corrar, f_mean, t_mean);

			corrar_max_index =
					max_index(corrar,
							((search_end - search_start)
									- (range_end - range_start) + 1));
			if ((corrar_max_index > 0)
					&& (corrar_max_index
							< ((search_end - search_start)
									- (range_end - range_start)))) {
				y1 = corrar[corrar_max_index];
				y0 = corrar[corrar_max_index - 1];
				y2 = corrar[corrar_max_index + 1];
			}

			if ((corrar_max_index == 0)) {
				y1 = corrar[corrar_max_index];
				y0 = y1;
				y2 = corrar[corrar_max_index + 1];
			}
			if ((corrar_max_index
					== ((search_end - search_start) - (range_end - range_start)))) {
				y1 = corrar[corrar_max_index];
				y0 = corrar[corrar_max_index - 1];
				y2 = y1;
			}

			w0_ratio = (y0 + y2) / (2 * y1);

			if (w0_ratio < -1 || w0_ratio > 1) {
				w0 = complex_acos((y0 + y2) / (2 * y1));
			} else {
				w0 = acos((y0 + y2) / (2 * y1));
			}

			theta = atan2((y0 - y2), (2 * y1 * sin(w0)));
			delta2 = -theta / w0;

			*(t + pt_index * uncomp_matrix_x_const * volume_size_const
					+ line_index) = (float) corrar_max_index + delta2;
			*(corr + pt_index * uncomp_matrix_x_const * volume_size_const
					+ line_index) = corrar[corrar_max_index];

			if (corrar[corrar_max_index] > CROSSCORRELATION_THRESHOLD_CUDA) {
				*(EI_1 + pt_index * uncomp_matrix_x_const * volume_size_const
						+ line_index) = *(t
						+ pt_index * uncomp_matrix_x_const * volume_size_const
						+ line_index) - max_strain_const / 2;
			} else {
				*(EI_1 + pt_index * uncomp_matrix_x_const * volume_size_const
						+ line_index) = 0;
			}
		}
	}
}


void set_threshold_values(float crosscorrelation_threshold,
		float negative_threshold_std_deviation,
		float negative_threshold_constant,
		float positive_threshold_std_deviation,
		float positive_threshold_constant, float strain_value_negative_noise,
		float strain_value_positive_noise) {
	CROSSCORRELATION_THRESHOLD = (float) (crosscorrelation_threshold / 1000.0);
	NEGATIVE_THRESHOLD_STD_DEVIATION = (float) (negative_threshold_std_deviation
			/ 1000.0);
	NEGATIVE_THRESHOLD_CONSTANT =
			(float) (negative_threshold_constant / 1000.0);
	POSITIVE_THRESHOLD_STD_DEVIATION = (float) (positive_threshold_std_deviation
			/ 1000.0);
	POSITIVE_THRESHOLD_CONSTANT =
			(float) (positive_threshold_constant / 1000.0);
	STRAIN_VALUE_NEGATIVE_NOISE =
			(float) (strain_value_negative_noise / 10000.0);
	STRAIN_VALUE_POSITIVE_NOISE =
			(float) (strain_value_positive_noise / 10000.0);

	cudaMemcpyToSymbol(CROSSCORRELATION_THRESHOLD_CUDA,
			&CROSSCORRELATION_THRESHOLD, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	checkCUDAError("CudaMemcpyToSymbol");
	cudaMemcpyToSymbol(NEGATIVE_THRESHOLD_STD_DEVIATION_CUDA,
			&NEGATIVE_THRESHOLD_STD_DEVIATION, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(NEGATIVE_THRESHOLD_CONSTANT_CUDA,
			&NEGATIVE_THRESHOLD_CONSTANT, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(POSITIVE_THRESHOLD_STD_DEVIATION_CUDA,
			&POSITIVE_THRESHOLD_STD_DEVIATION, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(POSITIVE_THRESHOLD_CONSTANT_CUDA,
			&POSITIVE_THRESHOLD_CONSTANT, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(STRAIN_VALUE_NEGATIVE_NOISE_CUDA,
			&STRAIN_VALUE_NEGATIVE_NOISE, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(STRAIN_VALUE_POSITIVE_NOISE_CUDA,
			&STRAIN_VALUE_POSITIVE_NOISE, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	checkCUDAError("CudaMemcpyToSymbol");

#ifdef DEBUG_THRESHOLD_VALUES
	printf("CROSSCORRELATION_THRESHOLD %.3f\n", CROSSCORRELATION_THRESHOLD);
	printf("NEGATIVE_THRESHOLD_STD_DEVIATION %.3f\n",
			NEGATIVE_THRESHOLD_STD_DEVIATION);
	printf("NEGATIVE_THRESHOLD_CONSTANT %.3f\n", NEGATIVE_THRESHOLD_CONSTANT);
	printf("POSITIVE_THRESHOLD_STD_DEVIATION %.3f\n",
			POSITIVE_THRESHOLD_STD_DEVIATION);
	printf("STRAIN_VALUE_NEGATIVE_NOISE %.3f\n", STRAIN_VALUE_NEGATIVE_NOISE);
	printf("STRAIN_VALUE_POSITIVE_NOISE %.3f\n", STRAIN_VALUE_POSITIVE_NOISE);
#endif

}

