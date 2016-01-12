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
	birthdayAttack();
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
