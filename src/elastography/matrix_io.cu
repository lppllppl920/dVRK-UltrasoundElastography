/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.
 ***************************************************************************/

#include "matrix_io.h"

int read_matrix(FILE *fp, float *target, int M) {
    char input[10000];
    char* tk;
    int f = 10000;
    char *token;
    int i = 0;
    int j = 0;
    while (fgets(input, f, fp)) {
        i = 0;
        tk = input;
        while ((token = strtok(tk, " ")) != NULL) {
            *(target + i * M + j) = ((float) atof(token));
            //*(target + i  + j * N) = ((float)atof(token));
            tk = NULL;
            //printf ("%f ", *(target + i +j*N));
            i++;
        }
        j++;
    }
    return 0;
}

int count_matrix(FILE *fp, int *x, int *y) {
    char input[10000];
    char* tk;
    int f = 10000;
    int i = 0;
    int j = 0;
    while (fgets(input, f, fp)) {
        i = 0;
        tk = input;
        while ((strtok(tk, " ")) != NULL) {
            tk = NULL;
            i++; //width
        }
        j++; //height
    }
    *y = j; //height
    *x = i; //width
    return 0;
}

int print_matrix_2(FILE *fp, float *target, int x, int y, int N) {
    int i, j;
    for (i = 0; i < x; i++) {
        //   printf("[%d]",i);
        for (j = 0; j < y; j++) {
            //printf("(%d,%f) ", j,*(target + i * N + j));
            printf("%f ", *(target + i * N + j));
        }
    }
    return 0;
}

int print_matrix_3(FILE *fp, short int *target, int height, int width,
        int N_width) {
    int i, j;
    for (i = 0; i < height; i++) {
        //   printf("[%d]",i);
        for (j = 0; j < width; j++) {
            //printf("(%d,%f) ", j,*(target + i * N + j));
            printf("%d ", *(target + i * N_width + j));
        }
        printf("\n");
    }
    return 0;
}

int print_matrix_4(int iteration_count, unsigned char *source, int height,
        int width, int N_width) {
    char file_name[100];
    FILE *fp;
    int i, j;
    sprintf(file_name, "c:\\ei_out\\output_%d.txt", iteration_count);
    fp = fopen(file_name, "w");

    for (i = 0; i < height; i++) {
        for (j = 0; j < width - 1; j++) {
            fprintf(fp, "%d ", (int) source[i * N_width + j]);
        }
        fprintf(fp, "%d\n", (int) source[i * N_width + j]);
    }
    fclose(fp);
    return 0;
}
double linear_interpolate(double x1, double x2, double x, double a, double b) {
    return fabs(x - x1) / fabs(x2 - x1) * b + fabs(x2 - x) / fabs(x2 - x1) * a;
}
double get_bilinear_sample(unsigned char *strain, double x, double y, int width,
        int height) {
    int y_upper;
    int y_lower;
    int x_left;
    int x_right;

    double top;
    double lower;
    double temp;

    y_upper = round(y + 0.5);
    y_lower = floor(y);

    x_left = floor(x);
    x_right = round(x + 0.5);

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
        return strain[x_left + y_upper * width];
    }

    if (y_upper == y_lower) {
        //TODO scan conversion in x direction
        return linear_interpolate(x_left, x_right, x,
                strain[x_left + y_upper * width],
                strain[x_right + y_upper * width]);
    }

    if (x_left == x_right) {
        //TODO scan conversion in y direction
        return linear_interpolate(y_lower, y_upper, y,
                strain[x_left + y_lower * width],
                strain[x_right + y_upper * width]);
    }

    top = linear_interpolate(x_left, x_right, x,
            strain[x_left + y_upper * width],
            strain[x_right + y_upper * width]);
    lower = linear_interpolate(x_left, x_right, x,
            strain[x_left + y_lower * width],
            strain[x_right + y_lower * width]);

    temp = linear_interpolate(y_lower, y_upper, y, lower, top);

    return (unsigned char) temp;
}

double get_nearest_sample(unsigned char *strain, double x, double y, int width,
        int height) {
    int local_x = round(x);
    int local_y = round(y);
    if (local_x >= width) {
        local_x = width - 1;
    }

    if (local_y >= height) {
        local_y = height - 1;
    }

    return strain[local_y * width + local_x];
}

void scale_image(unsigned char *out_strain, int new_width, int new_height,
        unsigned char *strain, int width, int height, int algorithm) {
    int i, j;
    double temp;
    double height_scale;
    double width_scale;

    height_scale = (double) new_height / height;
    width_scale = (double) new_width / width;

    for (j = 0; j < new_height; j++) {
        for (i = 0; i < new_width; i++) {
            if (algorithm == 0) {
                out_strain[j * new_width + i] = (unsigned char) round(
                        get_bilinear_sample(strain, i / width_scale,
                                j / height_scale, width, height));
            } else {
                out_strain[j * new_width + i] = (unsigned char) round(
                        get_nearest_sample(strain, i / width_scale,
                                j / height_scale, width, height));
            }
        }
    }
}

void scale_image_mm(unsigned char **out_Im1, int *out_width, int *out_height,
        unsigned char *Im1, int width, int height, int NOF, double spacing_x,
        double spacing_y) {
    int new_height;
    int new_width;
    int i;

    unsigned char *Im2;

    double width_mm = width * spacing_x;
    double height_mm = height * spacing_y;

    new_width = width;
    new_height = (height_mm / width_mm) * new_width;

    Im2 = (unsigned char *) malloc(
            sizeof(unsigned char) * new_width * new_height * NOF);

    if (Im2 == NULL) {
        printf("error allocating memory for Im2 width %d height %d\n",
                new_width, new_height);
        exit(1);
    }

    for (i = 0; i < NOF; i++) {
        scale_image(Im2 + i * new_width * new_height, new_width, new_height,
                Im1 + i * width * height, width, height, 0);
    }

    *out_width = new_width;
    *out_height = new_height;
    *out_Im1 = Im2;
}
