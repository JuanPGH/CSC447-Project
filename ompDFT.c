// Code Adapted from: https://www.math.wustl.edu/~victor/mfmm/fourier/fft.c & https://www.geeksforgeeks.org/discrete-fourier-transform-and-its-inverse-using-c/

// GNUPLOT was only used in the OMP version since the MPI version was run on a virtual machine and the CUDA version was run through Google Colab

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <omp.h>
#include <time.h>

#define q 14
#define N (int)pow(2,q)

#define OMPTHREADS 8

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

void ompdft(complex* v, int n, complex* tmp) {
	printf("OpenMP - DFT...\n\n");
	omp_set_num_threads(OMPTHREADS);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int i;
		printf("OMP Thread %d Running...\n", id);
		#pragma omp for
		for (i = 0; i < n; i++) {
			tmp[i].Re = 0.0;
			tmp[i].Im = 0.0;
			for (int j = 0; j < n; j++) {
				tmp[i].Re = 
					(tmp[i].Re + v[j].Re * cos(2 * PI * i * j / n));
				tmp[i].Im =
					(tmp[i].Im - v[j].Re * sin(2 * PI * i * j / n));
			}
		}
		printf("OMP Thread %d Finished.\n", id);
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
		sample = sin(20*x) + cos(200*x);
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

void main() {
	// Allocate working and scratch arrays for complex numbers
	complex* v;
	v = malloc(sizeof(complex) * N);
	complex* scratch;
	scratch = malloc(sizeof(complex) * N);
	if (v == NULL || scratch == NULL)
		exit(-1);

	// Declare file pointers and args for gnuplot and data
	FILE* fp = NULL;
	FILE* gnuPipe = NULL;
	int num;
	char* gnuArgs1[] = {
		"plot \"input.dat\" w lines",
	};
	char* gnuArgs2[] = {
		"plot \"fft.dat\" w lines",
	};

	// Fill working array with complex numbers and display
	fillInput(v, N);
	if (N<17) 
		print_vector("Input", v, N);

	// Copy numbers to file "input.dat"
	fopen_s(&fp, "input.dat", "w");
	if (fp == NULL)
		exit(-1);
	for (int i = 0; i < N; i++) {
		fprintf(fp, "%d %.2f\n", i, v[i].Re);
	}
	fclose(fp);

	// Perform the dft on the input
	omp_set_num_threads(OMPTHREADS);
	clock_t start, end;
	double cpu_time;
	start = clock();
	ompdft(v, N, scratch);
	end = clock();
	if (N < 17)
		print_vector("DFT", v, N);
	cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("DFT: Time taken is %.3f\n",cpu_time);

	// Copy numbers to file "fft.dat"
	fopen_s(&fp, "fft.dat", "w");
	if (fp == NULL)
		exit(-1);
	double temp;
	for (int i = 0; i < N / 2; i++) {
		double re, im;
		re = v[i].Re;
		im = v[i].Im;
		temp = 20 * log10(sqrt(re * re + im * im));
		fprintf(fp, "%d %.2f\n", i, temp);
	}
	fclose(fp);

	// Pipe args to command line and plot "input.dat"
	gnuPipe = _popen("gnuplot -persistent", "w");
	num = sizeof(gnuArgs1) / sizeof(gnuArgs1[0]);
	for (int i = 0; i < num; i++) { // 2 is size of gnuArgs
		fprintf(gnuPipe, "%s\n", gnuArgs1[i]);
	}
	fclose(gnuPipe);

	// Pipe args to command line and plot "fft.dat"
	gnuPipe = _popen("gnuplot -persistent", "w");
	num = sizeof(gnuArgs2) / sizeof(gnuArgs2[0]);
	for (int i = 0; i < num; i++) { // 2 is size of gnuArgs
		fprintf(gnuPipe, "%s\n", gnuArgs2[i]);
	}
	fclose(gnuPipe);


	// Deallocate arrays	
	free(v);
	free(scratch);
}