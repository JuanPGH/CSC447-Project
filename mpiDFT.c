// Code Adapted from: https://www.math.wustl.edu/~victor/mfmm/fourier/fft.c & https://www.geeksforgeeks.org/discrete-fourier-transform-and-its-inverse-using-c/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "mpi.h"

#define q 14
#define N (int)pow(2,q)

typedef struct {
    double Re;
    double Im;
} complex;

#define PI 3.14159265358979323846264338327950288

void print_vector(const char* title, complex* x, int n) {
    int i;
    printf("%s (dim=%d):", title, n);
    for (i = 0; i < n; i++) printf(" %5.2f,%5.2f ", x[i].Re, x[i].Im);
    putchar('\n');
    return;
}

void dft(complex* v, int n, complex* tmp) {
    for (int i = 0; i < n; i++) {
        tmp[i].Re = 0.0;
        tmp[i].Im = 0.0;
        for (int j = 0; j < n; j++) {
            tmp[i].Re += v[j].Re * cos(2 * PI * i * j / n) + v[j].Im * sin(2 * PI * i * j / n);
            tmp[i].Im += v[j].Im * cos(2 * PI * i * j / n) - v[j].Re * sin(2 * PI * i * j / n);
        }
    }
    for (int i = 0; i < n; i++) {
        v[i].Re = tmp[i].Re;
        v[i].Im = tmp[i].Im;
    }
    return;
}

void fillInput(complex* v, int n) {
    int sampleNum = n;
    double length = 10.0;
    double sample;

    for (int i = 0; i < sampleNum; i++) {
        double x = i * (length / sampleNum);
        sample = sin(20 * x) + cos(200 * x);
        v[i].Re = (double)(sample);
        v[i].Im = 0.0;
    }

    if (sampleNum < 17) {
        for (int i = 0; i < sampleNum; i++) {
            printf("%.2f ", v[i].Re);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the current process group
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate working and scratch arrays for complex numbers
    complex* v;
    v = malloc(sizeof(complex) * N);
    complex* scratch;
    scratch = malloc(sizeof(complex) * N / size);
    if (v == NULL || scratch == NULL)
        exit(-1);

    // Fill working array with complex numbers and display
    double start_time, end_time;
    start_time = MPI_Wtime();
    if (rank == 0) {
        v = malloc(sizeof(complex) * N);
        fillInput(v, N);
    }
    MPI_Scatter(v, N / size, MPI_DOUBLE_COMPLEX, v,
	N / size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    
    // Perform DFT on local input array
    dft(v, N / size, scratch);

    // Gather results from all processes into the root process
    MPI_Gather(v, N / size, MPI_DOUBLE_COMPLEX, v,
	N / size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    // Display final output
     if (rank == 0) { 
        end_time = MPI_Wtime();
        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    // Free memory
    free(v);
    free(scratch);
    return 0;
}