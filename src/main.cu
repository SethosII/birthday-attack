// C standard header files
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA header files
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef struct Handle {
	bool gpu;
	bool help;
	bool verbose;
} Handle;

void fillRandomCPU(double* array, const int length);
__global__ void fillRandomGPU(double* array, const int length);
void printArray(const double* array, const int length);
void processParameters(Handle* handle, int argc, char* argv[]);

int main(int argc, char* argv[]) {
	Handle handle = { .gpu = false, .help = false, .verbose = false };
	processParameters(&handle, argc, argv);

	int randomNumbersLength = 10;
	size_t randomNumbersSize = randomNumbersLength * sizeof(double);
	double* randomNumbers = (double*) malloc(randomNumbersLength * sizeof(double));

	if (handle.gpu) {
		double* d_randomNumbers;
		cudaMalloc(&d_randomNumbers, randomNumbersSize);

		fillRandomGPU<<<1,1>>>(randomNumbers, randomNumbersLength);

		cudaMemcpy(randomNumbers, d_randomNumbers, randomNumbersSize, cudaMemcpyDeviceToHost);

		cudaFree(d_randomNumbers);
	} else {
		fillRandomCPU(randomNumbers, randomNumbersLength);
	}

	printArray(randomNumbers, randomNumbersLength);

	free(randomNumbers);
}

void fillRandomCPU(double* array, const int length) {
	for (int i = 0; i < length; i++) {
		array[i] = (double) rand() / RAND_MAX;
	}
}

__global__ void fillRandomGPU(double* array, const int length) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState state;
	curand_init(id, id, id, &state);
	printf("%d\n", curand(&state));
	for (int i = 0; i < length; i++) {
		array[i] = curand(&state);
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
