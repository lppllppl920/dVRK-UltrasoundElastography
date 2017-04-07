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

#include "scan_conversion.h"

void checkCUDAError(const char *msg);

void __global__ dummy ()
{

}

void cuda_malloc_host(void **ptr, size_t ptr_size);

void __global__ g_curved_scan_convert (data_type *buffer, data_type *out,
        int out_x, int out_y, int x, int y, int z,
        float center, float probeRad, float probeAng, float angStep, int start)
{
    /*    int i = blockIdx.y;//corresponds to y
     int j = blockIdx.x;//corresponds to x
     int k = threadIdx.x + start;*/

    int i = blockIdx.y; //corresponds to y
    int j = blockIdx.x;//corresponds to x
    int k = threadIdx.x + start;

    float r;
    float theta;
    int i2;
    int j2;

    out = out + k * out_x * out_y;
    buffer = buffer + k * x * y;

    r = probeRad + i + 1;
    theta = -probeAng/2 + j * angStep;
    i2 = round (r * cos (theta));
    j2 = center + round (r * sin (theta));
    out[(int) round((float)i2 * out_x + j2)] = buffer [i * x + j];
}

void __global__ g_curved_scan_convert_3 (data_type *buffer, data_type *out,
        int out_x, int out_y, int x, int y, int z,
        float center, float probeRad, float probeAng, float angStep, float cut_off, float ratio_x, float ratio_y, int start)
{
    int i = blockIdx.y; //corresponds to y
    int k = blockIdx.x;//corresponds to x
    int j = threadIdx.x + start;

    /*int i = blockIdx.y;//corresponds to y
     int j = blockIdx.x;//corresponds to x
     int k = threadIdx.x + start;*/

    float r;
    float theta;
    int i2;
    int j2;
    int i3;
    int j3;

    out = out + k * out_x * out_y;
    buffer = buffer + k * x * y;

    r = probeRad + i + 1;
    theta = -probeAng/2 + j * angStep;
    i2 = round (r * cos (theta));

    i2 = i2 - cut_off;

    i3 = i2 * ratio_y;

    j2 = center + round (r * sin (theta));

    j3 = j2 * ratio_x;

    if (buffer [i * x + j] > out[(int) round((float)i3 * out_x + j3)]) {
        out[(int) round((float)i3 * out_x + j3)] = buffer [i * x + j];
    }
}

void __global__ g_curved_scan_convert_2 (data_type *buffer, data_type *out,
        int out_x, int out_y, int x, int y, int z,
        float center, float probeRad, float probeAng, float angStep, int start)
{
    int i = blockIdx.y; //corresponds to y
    int k = blockIdx.x;//corresponds to x
    int j = threadIdx.x + start;

    /*int i = blockIdx.y;//corresponds to y
     int j = blockIdx.x;//corresponds to x
     int k = threadIdx.x + start;*/

    float r;
    float theta;
    int i2;
    int j2;

    out = out + k * out_x * out_y;
    buffer = buffer + k * x * y;

    r = probeRad + i + 1;
    theta = -probeAng/2 + j * angStep;
    i2 = round (r * cos (theta));
    j2 = center + round (r * sin (theta));
    out[(int) round((float)i2 * out_x + j2)] = buffer [i * x + j];
}

void __global__ g_rotate_image (data_type *output, data_type *input, int x, int y, int z) {
    int i, j, k;

    i = blockIdx.x;
    j = blockIdx.y;
    k = threadIdx.x;
    data_type temp;

    temp = input [(x-j-1) * y * z + k * z + i];

    input [(x-j-1) * y * z + k * z + i] =
    output [i * x * y + k * x + j];

    output [i * x * y + k * x + j] = temp;

}

void __global__ g_average_filter (data_type *output, data_type *buffer, int x,
        int y, int z, int window)
{
    /*int i = threadIdx.x;
     int j = blockIdx.y;
     int k = blockIdx.x;*/

    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

    int l, m;

    float sum = 0;
    int count = 0;

    for (l = -window; l <= window; l++) {
        if ((l + j) < 0) {
            continue;
        }
        if ((l+j) > (y-1)) {
            continue;
        }
        for (m = -window; m <= window; m++) {
            if ((m + k) < 0) {
                continue;
            }
            if ((m+k) > (x-1)) {
                break;
            }
            if (buffer [i * x * y + (l+j) * x + (m+k)] != 0) {
                sum += buffer [i * x * y + (l+j) * x + (m+k)];
                count ++;
            }
        }
    }

    if (count != 0) {
        output [i * x * y + j * x + k] = sum / count;
    } else {
        output [i * x * y + j * x + k] = 0;
    }
}

data_type * cuda_average_filter(data_type *buffer, int x, int y, int z,
        int window) {

    int size = sizeof(data_type) * x * y * z;

    data_type *output;

    checkCudaErrors(cudaMallocHost(&output, sizeof(data_type) * x * y * z));

    /*if (output == NULL) {
     printf ("Error allocating memory\n");
     exit(1);
     }*/

    data_type *output_d;
    data_type *buffer_d;

    checkCudaErrors(cudaMalloc(&output_d, size));
    checkCudaErrors(cudaMalloc(&buffer_d, size));

    checkCudaErrors(cudaMemcpy(buffer_d, buffer, size, cudaMemcpyHostToDevice));

    dim3 blocks = dim3(z, y);
    dim3 threads = dim3(x);

    g_average_filter <<<blocks, threads>>> (output_d, buffer_d, x, y, z,
            window);
    checkCUDAError("g_average_filter");
    checkCudaErrors(cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaFreeHost(buffer));
    free(buffer);

    checkCudaErrors(cudaFree(output_d));
    checkCudaErrors(cudaFree(buffer_d));

    return output;
}

void cuda_rotate_image(data_type *output, data_type *input, int x, int y,
        int z) {

    data_type *output_d;
    data_type *input_d;

    int size = sizeof(data_type) * x * y * z;

    checkCudaErrors(cudaMalloc(&output_d, size));
    checkCudaErrors(cudaMalloc(&input_d, size));

    checkCudaErrors(cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice));

    dim3 blocks = dim3(x, y);
    dim3 threads = dim3(z);

    g_rotate_image <<<blocks, threads>>> (output_d, input_d, x, y, z);

    checkCUDAError("g_rotate_image");

    checkCudaErrors(cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost));
    cudaThreadSynchronize();
    checkCudaErrors(cudaFree(output_d));
    checkCudaErrors(cudaFree(input_d));
}

void __global__ g_curved_scan_convert (data_type *buffer, data_type *out,
        int height, int width, int length, int x, int y, int z,
        int pixel_spacing_x, int pixel_spacing_y, int pixel_spacing_z,
        float pitch, float speed_of_sound, float sampling_frequency)
{
    /*int i = blockIdx.x;
     int j = blockIdx.y;
     int k = threadIdx.x;*/

    int i = threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.x;

    int X = ((float)k/pixel_spacing_x) * ((float)x / (y * pitch));
    int Z = ((float)i/pixel_spacing_z) * ((float)x / (y * pitch));
    int Y = ((float) j/pixel_spacing_y) *
    ((float)sampling_frequency/ speed_of_sound);

    out [i * height * width + j * width + k] =
    buffer [Z * x * y + Y * x + X];
}

void __global__ g_linear_scan_conversion (data_type *output, data_type *input,
        int x, int y, int z, int display_x, int display_y, int display_z,
        int width, int height)
{
    int i = threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.x;

    int Y = ((float) j * width / (float)height);

    if (Y>=y) {
        Y--;
    }

    int X = k;

    output [i * (display_x) * (display_y) + j * (display_x) + k]
    = input [i * x * y + Y * x + X];
}

int cuda_linear_scan_conversion(data_type *input, data_type **output_t, int x,
        int y, int z, int *display_x, int *display_y, int *display_z,
        float pitch, float speed_of_sound, float sampling_frequency,
        int number_scan_lines) {
    int output_size;

    data_type *output;

    data_type *output_d;

    data_type *input_d;

    int input_size = sizeof(data_type) * x * y * z;

    int width = x * pitch;

    int height = (float) y * speed_of_sound / (sampling_frequency * 2);

    printf("Linear scan convert width %d height %d\n", width, height);

    fflush(stdout);

    *display_x = 400;

    *display_y = ((float) height * (*display_x) / width);

    *display_z = z;

    output_size = sizeof(data_type) * (*display_x) * (*display_y)
            * (*display_z);

    //output = (data_type *) malloc (output_size);

    checkCudaErrors(cudaMallocHost(&output, output_size));

    /*if (output == NULL) {
     printf ("could not allocate memory\n");
     fflush (stdout);
     exit (1);
     } */

    printf("Output size %d\n", output_size);

    checkCudaErrors(cudaMalloc(&output_d, output_size));
    checkCudaErrors(cudaMalloc(&input_d, input_size));

    checkCudaErrors(
            cudaMemcpy(input_d, input, input_size, cudaMemcpyHostToDevice));

    dim3 blocks = dim3(*display_x, *display_y);
    dim3 threads = dim3(*display_z);

    g_linear_scan_conversion <<<blocks, threads>>>(output_d, input_d,
            x, y, z, *display_x, *display_y, *display_z, width, height);

    checkCUDAError("g_linear_scan_convert");

    checkCudaErrors(
            cudaMemcpy(output, output_d, output_size, cudaMemcpyDeviceToHost));
    *output_t = output;

    checkCudaErrors(cudaFree(output_d));
    checkCudaErrors(cudaFree(input_d));

    return 1;
}

int set_cuda_device(int device_id);

data_type * cuda_curved_scan_convert_2(data_type *buffer, int *out_x,
        int *out_y, int *out_z, int x, int y, int z, float radios, float ang,
        float spacing_y, float * final_x) {
    float probeAng = ang * 3.14 / 180;
    float probeRad = radios;

    //float x2 = round (probeRad + x);
    float y2 = round(probeRad + y);
    //float center = round (x2 * sin (probeAng/2));
    float center = round(y2 * sin(probeAng / 2));
    //float y2 = 2 * center;
    float x2 = 2 * center;

    float angStep = probeAng / x;
//0.75 is the width at 0.75 x y point of the original image
    int new_x = x2 * (probeRad + 0.75 * y) * sin(angStep) * spacing_y;

    *final_x = (probeRad + 0.75 * y) * sin(angStep) * spacing_y;

    //int new_x = x2 * (probeRad + y) * sin (angStep) * spacing_y;

    //new_x = 200;

    printf("sin_theta %f new_x %d\n", sin(angStep), new_x);
    int new_y;

    float cut_off;

    data_type *out;
    int output_size;
    data_type *out_local;
    data_type *buffer_d;
    int input_size;

    cut_off = probeRad * cos(probeAng / 2) - 0;

    int i_y2 = round(y2 - cut_off);
    int i_x2 = round(x2);

    new_y = new_x * (float) i_y2 / (float) i_x2;

    input_size = sizeof(data_type) * x * y * z;

    checkCudaErrors(cudaMalloc(&buffer_d, input_size));

    checkCudaErrors(
            cudaMemcpy(buffer_d, buffer, input_size, cudaMemcpyHostToDevice));

    output_size = sizeof(data_type) * new_x * new_y * z;

    checkCudaErrors(cudaMalloc(&out, output_size));

    checkCudaErrors(cudaMemset(out, 0, output_size));

    out_local = (data_type *) malloc(output_size);

    // checkCudaErrors(cudaMallocHost (&out_local, output_size));

    if (out_local == NULL) {
        printf("Could not allocate memory for out\n");
        exit(1);
    }

    dim3 blocks = dim3(z, y, 1);
    dim3 threads = dim3(x, 1, 1);

    /*dim3 blocks = dim3(z, y, 1);
     dim3 threads = dim3 (x, 1, 1);*/
    checkCUDAError("g_curved_scan_convert 2");

    printf("%d\n", x * y);
    dummy <<<blocks, threads>>> ();

    checkCUDAError("DUMMY");

    printf("new_x %d new_y %d y2 %f x2 %f i_y2 %d i_x2 %d output_size %d\n",
            new_x, new_y, y2, x2, i_y2, i_x2, output_size);
    fflush(stdout);

    float ratio_x = (float) new_x / (float) i_x2;
    float ratio_y = (float) new_y / (float) i_y2;
    g_curved_scan_convert_3 <<<blocks, threads>>> (buffer_d, out, new_x, new_y,
            x, y, z, center, probeRad, probeAng, angStep, cut_off, ratio_x, ratio_y, 0);

    checkCUDAError("g_scan_convert 3");

    checkCudaErrors(
            cudaMemcpy(out_local, out, output_size, cudaMemcpyDeviceToHost));

    *out_x = new_x;
    *out_y = new_y;
    *out_z = z;
    checkCudaErrors(cudaFree(buffer_d));
    checkCudaErrors(cudaFree(out));
    return out_local;
}

data_type * cuda_curved_scan_convert_2_old(data_type *buffer, int *out_x,
        int *out_y, int *out_z, int x, int y, int z, float radios, float ang) {
    float probeAng = ang * 3.14 / 180;
    float probeRad = radios;

    //float x2 = round (probeRad + x);
    float y2 = round(probeRad + y);
    //float center = round (x2 * sin (probeAng/2));
    float center = round(y2 * sin(probeAng / 2));
    //float y2 = 2 * center;
    float x2 = 2 * center;

    float angStep = probeAng / x;

    data_type *out;
    int output_size;
    data_type *out_local;
    data_type *buffer_d;
    int input_size;

    int i_y2 = round(y2);
    int i_x2 = round(x2);

    input_size = sizeof(data_type) * x * y * z;

    checkCudaErrors(cudaMalloc(&buffer_d, input_size));

    checkCudaErrors(
            cudaMemcpy(buffer_d, buffer, input_size, cudaMemcpyHostToDevice));

    output_size = sizeof(data_type) * i_x2 * i_y2 * z;

    checkCudaErrors(cudaMalloc(&out, output_size));

    checkCudaErrors(cudaMemset(out, 0, output_size));

    //out_local = (data_type *) malloc (output_size);

    checkCudaErrors(cudaMallocHost(&out_local, output_size));

    /*if (out_local == NULL) {
     printf ("Could not allocate memory for out\n");
     exit (1);
     }*/

    dim3 blocks = dim3(z, y, 1);
    dim3 threads = dim3(x, 1, 1);

    /*dim3 blocks = dim3(z, y, 1);
     dim3 threads = dim3 (x, 1, 1);*/
    checkCUDAError("g_curved_scan_convert 1");

    printf("%d\n", x * y);
    dummy <<<blocks, z>>> ();

    checkCUDAError("DUMMY");

    g_curved_scan_convert_2 <<<blocks, threads>>> (buffer_d, out, i_x2, i_y2,
            x, y, z, center, probeRad, probeAng, angStep, 0);

    checkCUDAError("g_scan_convert 3");

    checkCudaErrors(
            cudaMemcpy(out_local, out, output_size, cudaMemcpyDeviceToHost));

    *out_x = i_x2;
    *out_y = i_y2;
    *out_z = z;
    checkCudaErrors(cudaFree(buffer_d));
    checkCudaErrors(cudaFree(out));
    return out_local;
}

data_type * cuda_curved_scan_convert(data_type *buffer, int *out_x, int *out_y,
        int *out_z, int x, int y, int z, float radios, float ang) {
    float probeAng = ang * 3.14 / 180;
    float probeRad = radios;

    //float x2 = round (probeRad + x);
    float y2 = round(probeRad + y);
    //float center = round (x2 * sin (probeAng/2));
    float center = round(y2 * sin(probeAng / 2));
    //float y2 = 2 * center;
    float x2 = 2 * center;

    float angStep = probeAng / x;

    data_type *out;
    int output_size;
    data_type *out_local;
    data_type *buffer_d;
    int input_size;

    int i_y2 = round(y2);
    int i_x2 = round(x2);

    input_size = sizeof(data_type) * x * y * z;

    checkCudaErrors(cudaMalloc(&buffer_d, input_size));

    checkCudaErrors(
            cudaMemcpy(buffer_d, buffer, input_size, cudaMemcpyHostToDevice));

    output_size = sizeof(data_type) * i_x2 * i_y2 * z;

    checkCudaErrors(cudaMalloc(&out, output_size));

    checkCudaErrors(cudaMemset(out, 0, output_size));

    //out_local = (data_type *) malloc (output_size);

    checkCudaErrors(cudaMallocHost(&out_local, output_size));

    /*if (out_local == NULL) {
     printf ("Could not allocate memory for out\n");
     exit (1);
     }*/

    dim3 blocks = dim3(x, y, 1);
    dim3 threads = dim3(z, 1, 1);

    /*dim3 blocks = dim3(z, y, 1);
     dim3 threads = dim3 (x, 1, 1);*/
    checkCUDAError("g_curved_scan_convert 1");

    printf("%d\n", x * y);
    dummy <<<blocks, z>>> ();

    checkCUDAError("DUMMY");

    g_curved_scan_convert <<<blocks, threads>>> (buffer_d, out, i_x2, i_y2,
            x, y, z, center, probeRad, probeAng, angStep, 0);

    checkCUDAError("g_scan_convert 3");

    checkCudaErrors(
            cudaMemcpy(out_local, out, output_size, cudaMemcpyDeviceToHost));

    *out_x = i_x2;
    *out_y = i_y2;
    *out_z = z;
    checkCudaErrors(cudaFree(buffer_d));
    checkCudaErrors(cudaFree(out));
    return out_local;
}

__device__ data_type_out d_linear_interpolate(double x1, double x2, double x,
        data_type_out a, data_type_out b) {
    return (data_type_out) round(
            fabs(x - x1) / fabs(x2 - x1) * b + fabs(x2 - x) / fabs(x2 - x1) * a);
}

__device__ data_type_out d_get_bilinear_sample(data_type_out *strain, double x,
        double y, int z, int width, int height) {
    int y_upper;
    int y_lower;
    int x_left;
    int x_right;

    data_type_out top;
    data_type_out lower;
    data_type_out temp;

    y_upper = (int) round(y + 0.5);
    y_lower = (int) floor(y);

    x_left = (int) floor(x);
    x_right = (int) round(x + 0.5);

    if (y_upper >= height) {
        y_upper = height - 1;
    }

    if (x_right >= width) {
        x_right = width - 1;
    }

    if (y_lower >= height) {
        y_lower = height - 1;
    }

    if (x_left >= width) {
        x_left = width - 1;
    }

    if ((y_upper == y_lower) && (x_left == x_right)) {
        return strain[x_left + y_upper * width + z * width * height];
    }

    if (y_upper == y_lower) {
        //TODO scan conversion in x direction
        return d_linear_interpolate(x_left, x_right, x,
                strain[x_left + y_upper * width + z * width * height],
                strain[x_right + y_upper * width + z * width * height]);
    }

    if (x_left == x_right) {
        //TODO scan conversion in y direction
        return d_linear_interpolate(y_lower, y_upper, y,
                strain[x_left + y_lower * width],
                strain[x_right + y_upper * width]);
    }

    top = d_linear_interpolate(x_left, x_right, x,
            strain[x_left + y_upper * width],
            strain[x_right + y_upper * width]);
    lower = d_linear_interpolate(x_left, x_right, x,
            strain[x_left + y_lower * width],
            strain[x_right + y_lower * width]);

    temp = d_linear_interpolate(y_lower, y_upper, y, lower, top);

    return temp;
}

__device__ data_type_out d_get_nearest_sample(data_type_out *strain, double x,
        double y, int z, int width, int height) {
    int local_x = (int) round(x);
    int local_y = (int) round(y);
    if (local_x >= width) {
        local_x = width - 1;
    }

    if (local_y >= height) {
        local_y = height - 1;
    }

    return strain[local_y * width + local_x + z * width * height];
}

__global__ void g_scale_image(data_type_out *out_strain, int new_width,
        int new_height, data_type_out *strain, int width, int height,
        int no_of_frames, int algorithm) {
    int i, j, k;
    double height_scale;
    double width_scale;

    k = blockIdx.y;
    j = blockIdx.x;
    i = threadIdx.x;

    height_scale = (double) new_height / height;
    width_scale = (double) new_width / width;

    if (algorithm == 0) {
        out_strain[k * new_width * new_height + j * new_width + i] =
                d_get_bilinear_sample(strain + k * width * height,
                        i / width_scale, j / height_scale, 0, width, height);
    } else {
        out_strain[k * new_width * new_height + j * new_width + i] =
                d_get_nearest_sample(strain + k * width * height,
                        i / width_scale, j / height_scale, 0, width, height);
    }

}

void cuda_scale_image_mm(data_type_out **out_Im1, int *out_width,
        int *out_height, data_type_out *Im1, int width, int height,
        int no_of_frames, double spacing_x, double spacing_y, float &time1) {
    int new_height;
    int new_width;

    data_type_out *Im2;
    data_type_out *Im2_d;
    data_type_out *Im1_d;

    time1 = 0;

    double width_mm = width * spacing_x;
    double height_mm = height * spacing_y;

#ifdef CUDA_EVENT_TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
#endif

#ifdef CUDA_EVENT_TIMER
    cudaEventRecord(start, 0);
#endif

    new_width = width;
    new_height = (int) ((float) ((height_mm / width_mm) * (float) new_width));
    /*new_height = height;
     new_width = (int)((float)((width_mm/height_mm) * (float)new_height));*/

    int output_size = sizeof(data_type_out) * new_width * new_height
            * no_of_frames;
    int input_size = sizeof(data_type_out) * width * height * no_of_frames;

    Im2 = (data_type_out *) malloc(output_size);

    if (Im2 == NULL) {
        printf("error allocating memory for Im2 width %d height %d\n",
                new_width, new_height);
        exit(1);
    }

    checkCudaErrors(cudaMalloc(&Im2_d, output_size));

    checkCudaErrors(cudaMemset(Im2_d, 0, output_size));

    checkCudaErrors(cudaMalloc(&Im1_d, input_size));

    checkCudaErrors(cudaMemcpy(Im1_d, Im1, input_size, cudaMemcpyHostToDevice));

#ifdef CUDA_EVENT_TIMER
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for initial allocation of rectangular scan conversion: %f ms\n", time);
    time1 += time;
#endif   

    dim3 blocks(new_height, no_of_frames, 1);
    dim3 threads(new_width);

#ifdef CUDA_EVENT_TIMER
    cudaEventRecord(start, 0);
#endif

    g_scale_image <<<blocks, threads>>> (Im2_d, new_width, new_height, Im1_d, width, height, no_of_frames, 0);

    checkCUDAError("g_scale_image");

#ifdef CUDA_EVENT_TIMER
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for rectangular scan conversion g_scale_image: %f ms\n", time);
    time1 += time;
#endif  

#ifdef CUDA_EVENT_TIMER
    cudaEventRecord(start, 0);
#endif

    checkCudaErrors(
            cudaMemcpy(Im2, Im2_d, output_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(Im2_d));
    checkCudaErrors(cudaFree(Im1_d));
#ifdef CUDA_EVENT_TIMER
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for remainder part of rectangular scan conversion: %f ms\n", time);
    time1 += time;
#endif  

    *out_width = new_width;
    *out_height = new_height;
    *out_Im1 = Im2;
}

