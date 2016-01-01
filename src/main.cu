//#define NDEBUG // include to remove asserts

// C standard header files
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA header files
#include <cuda_runtime.h>
#include <curand_kernel.h>

// own header files
#include "sha256.h"

const int LENGHT = 10;
const size_t SIZE_OF_RANDS = LENGHT * sizeof(double);

typedef struct Handle {
	bool gpu;bool help;bool verbose;
} Handle;

void fillRandomCPU(double* array, const int length);
__global__ void fillRandomGPU(unsigned int seed, double* array,
		const int length);
void printArray(const double* array, const int length);
void processParameters(Handle* handle, int argc, char* argv[]);

int main(int argc, char* argv[]) {
	Handle
	handle = {.gpu = false, .help = false, .verbose = false};
	processParameters(&handle, argc, argv);

	double* randomNumbers = (double*) malloc(SIZE_OF_RANDS);

	if (handle.gpu) {
		double* d_randomNumbers;
		cudaMalloc((void **) &d_randomNumbers, SIZE_OF_RANDS);

		fillRandomGPU<<<1, 1>>>(time(NULL), d_randomNumbers, LENGHT);

		cudaMemcpy(randomNumbers, d_randomNumbers, SIZE_OF_RANDS,
				cudaMemcpyDeviceToHost);

		cudaFree(d_randomNumbers);
	} else {
		fillRandomCPU(randomNumbers, LENGHT);
	}

	printArray(randomNumbers, LENGHT);

	free(randomNumbers);

	if (handle.gpu) {
		printf("\nHash on GPU not ready yet\n");
	} else {
#ifndef NDEBUG
		testSha256LongInput();
		testReduceSha256();
#endif

		unsigned int constantTextLength = 3;
		unsigned char *constantText[] = { (unsigned char*) "This",
				(unsigned char*) "a", (unsigned char*) "day." };
		unsigned int stencilLength = 2;
		unsigned char *stencil[][2] = { { (unsigned char*) "is",
				(unsigned char*) "was" }, { (unsigned char*) "great",
				(unsigned char*) "bad" } };

		unsigned int maximalLength = constantTextLength + stencilLength;
		for (int i = 0; i < constantTextLength; i++) {
			maximalLength += strlen((char*) constantText[i]);
		}
		for (int i = 0; i < stencilLength; i++) {
			maximalLength +=
					(strlen((char*) stencil[i][0])
							> strlen((char*) stencil[i][1])) ?
							strlen((char*) stencil[i][0]) :
							strlen((char*) stencil[i][1]);
		}
		unsigned char combined[maximalLength];

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				sprintf((char*) combined, "%s %s %s %s %s", constantText[0],
						stencil[0][i], constantText[1], stencil[1][j],
						constantText[2]);
				printf("%s\n", combined);

				unsigned char hash[4];
				reducedHash(combined, hash, 1);
				printHash(hash, 4);
				printf("\n");
			}
		}
	}
}

void fillRandomCPU(double* array, const int length) {
	for (int i = 0; i < length; i++) {
		array[i] = (double) rand() / RAND_MAX;
	}
}

__global__ void fillRandomGPU(unsigned int seed, double* result,
		const int lenght) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(seed, id, id, &state);
	for (int i = 0; i < lenght; i++) {
		result[i] = curand(&state);
	}
}

void printArray(const double* array, const int length) {
	for (int i = 0; i < length; i++) {
		printf("array[%d] = %f\n", i, array[i]);
	}
}

/*
 * process the command line parameters and return a Handle struct with them
 */
void processParameters(Handle* handle, int argc, char* argv[]) {
	for (int currentArgument = 1; currentArgument < argc; currentArgument++) {
		switch (argv[currentArgument][1]) {
		case 'g':
			// switch to GPU version
			handle->gpu = true;
			break;
		case 'h':
			// print help message
			printf(
					"Parameters:\n"
							"\t-g\t\trun on the GPU\n"
							"\t-h\t\tprint this help message\n"
							"\t-v\t\tprint more information\n"
							"\nThis program is distributed under the terms of the LGPLv3 license\n");
			handle->help = true;
			exit(EXIT_SUCCESS);
			break;
		case 'v':
			// print more information
			handle->verbose = true;
			break;
		default:
			fprintf(stderr, "Wrong parameter: %s\n"
					"-h for help\n", argv[currentArgument]);
			exit(EXIT_FAILURE);
		}
	}
}
