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

// own header files
#include "sha256.h"
#include "birthdayAttack.h"

typedef struct Handle {
	bool gpu;bool help;bool verbose;
} Handle;

void hashtestCPU();
__global__ void hashtestGPU();
void processParameters(Handle* handle, int argc, char* argv[]);

int main(int argc, char* argv[]) {
	Handle
	handle = {.gpu = false, .help = false, .verbose = false};
	processParameters(&handle, argc, argv);


#ifndef NDEBUG
	if (handle.gpu) {
		hashtestGPU<<<1, 1>>>();
		cudaDeviceReset();
	} else {
		hashtestCPU();
	}
#endif

	unsigned int dim = pow(2, 16);

	unsigned char* d_hashs;
	cudaMalloc((void**) &d_hashs, dim * 4 * sizeof(unsigned char));

	dim3 blockDim(256);
	dim3 gridDim((dim + blockDim.x - 1) / blockDim.x);

	cudaEvent_t custart, custop;
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);
	cudaEventRecord(custart, 0);
	birthdayAttack<<<gridDim, blockDim>>>(d_hashs, dim);
	cudaEventRecord(custop, 0);
	cudaEventSynchronize(custop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, custart, custop);
	printf("birthdayAttack: %3.1f ms\n", elapsedTime);
	cudaEventDestroy(custart);
	cudaEventDestroy(custop);

	unsigned char hashs[8];
	cudaMemcpy(hashs, d_hashs, 2 * 4 * sizeof(unsigned char),
			cudaMemcpyDeviceToHost);
	printHash(&hashs[0], 4);
	printHash(&hashs[4], 4);

	cudaFree(d_hashs);
}

void hashtestCPU() {
	testSha256LongInput();
	testReduceSha256();
}

__global__ void hashtestGPU() {
	testSha256LongInput();
	testReduceSha256();
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

