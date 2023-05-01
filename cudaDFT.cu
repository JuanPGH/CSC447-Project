// Code Adapted from: https://www.math.wustl.edu/~victor/mfmm/fourier/fft.c & https://www.geeksforgeeks.org/discrete-fourier-transform-and-its-inverse-using-c/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define q 15
#define N (int)pow(2,q)
#define BLOCK_SIZE 256

typedef struct {
	double Re;
	double Im;
} complex;

#define PI	3.14159265358979323846264338327950288

void print_vector(const char* title, complex* x, int n) {
	int i;
	printf("%s (dim=%d):", title, n);
	for (i = 0; i < n; i++) printf(" %5.2f,%5.2f ", x[i].Re, x[i].Im);
	putchar('\n');
	return;
}

__global__ void dft_kernel(complex* v, int n, complex* tmp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tmp[i].Re = 0.0;
        tmp[i].Im = 0.0;
        for (int j = 0; j < n; j++) {
            tmp[i].Re += v[j].Re * cos(2 * M_PI * i * j / n) + 
				v[j].Im * sin(2 * M_PI * i * j / n);
            tmp[i].Im += -v[j].Re * sin(2 * M_PI * i * j / n) + 
				v[j].Im * cos(2 * M_PI * i * j / n);
        }
    }
}

void cudft(complex* v, int n, complex* tmp) {
    complex* dev_v;
    complex* dev_tmp;

    cudaMalloc((void**)&dev_v, n * sizeof(complex));
    cudaMalloc((void**)&dev_tmp, n * sizeof(complex));

    cudaMemcpy(dev_v, v, n * sizeof(complex), cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Number of blocks used: %d\n", num_blocks);
    dft_kernel<<<num_blocks, BLOCK_SIZE>>>(dev_v, n, dev_tmp);

    cudaMemcpy(tmp, dev_tmp, n * sizeof(complex), cudaMemcpyDeviceToHost);

    cudaFree(dev_v);
    cudaFree(dev_tmp);

    for (int i = 0; i < n; i++) {
        v[i].Re = tmp[i].Re;
        v[i].Im = tmp[i].Im;
    }
}

void dft(complex* v, int n, complex* tmp) {
	for (int i = 0; i < n; i++) {
		tmp[i].Re = 0.0;
		tmp[i].Im = 0.0;
		for (int j = 0; j < n; j++) {
			tmp[i].Re = (tmp[i].Re + v[j].Re * cos(2 * PI * i * j / n));
			tmp[i].Im = (tmp[i].Im - v[j].Re * sin(2 * PI * i * j / n));
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

int main() {
	// Allocate working and scratch arrays for complex numbers
	complex* v;
	v = (complex*)malloc(sizeof(complex) * N);
	complex* scratch;
	scratch = (complex*)malloc(sizeof(complex) * N);
	if (v == NULL || scratch == NULL)
		exit(-1);

	// Fill working array with complex numbers and display
	fillInput(v, N);
	if (N < 17)
		print_vector("Input", v, N);

	// Perform the dft on the input
	clock_t start, end;
	double cpu_time;
	start = clock();
	cudft(v, N, scratch);
	end = clock();
	if (N < 17)
		print_vector("DFT", v, N);
	cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("DFT: Time taken is %.3f\n", cpu_time);

	// Deallocate arrays	
	free(v);
	free(scratch);

  return 0;
}