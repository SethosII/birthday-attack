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

__global__ void birthdayAttack(unsigned char* hashs, unsigned int dim);
void fillRandomCPU(double* array, const int length);
__global__ void fillRandomGPU(unsigned int seed, double* array,
		const int length);
__global__ void hashtestGPU();
void printArray(const double* array, const int length);
void processParameters(Handle* handle, int argc, char* argv[]);

__managed__ unsigned char* goodText = (unsigned char*) ".................";
__managed__ unsigned int goodTextOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17 };
__managed__ unsigned char* goodStencil =
		(unsigned char*) "qwertzuiopasdfghjklyxcvbnm123456";
__managed__ unsigned int goodStencilOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
		28, 29, 30, 31, 32 };

__managed__ unsigned char* badText = (unsigned char*) ".................";
__managed__ unsigned int badTextOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17 };
__managed__ unsigned char* badStencil =
		(unsigned char*) "qqwweerrttzzuuiiooppaassddffgghhjjkkllyyxxccvvbbnnmm112233445566";
__managed__ unsigned int badStencilOffsets[] = { 0, 2, 4, 6, 8, 10, 12, 14, 16,
		18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
		54, 56, 58, 60, 62, 64 };

__managed__ bool collision = false;

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
		hashtestGPU<<<1, 1>>>();
		cudaDeviceReset();
	} else {
#ifndef NDEBUG
		testSha256LongInput();
		testReduceSha256();
#endif

		unsigned int constantTextLength = 3;
		unsigned char *constantText[] = { (unsigned char*) "This",
				(unsigned char*) "a", (unsigned char*) "day." };
		unsigned int stencilLength = 4;
		unsigned char *stencil[] = { (unsigned char*) "is",
				(unsigned char*) "was", (unsigned char*) "great",
				(unsigned char*) "bad" };

		unsigned int maximalLength = constantTextLength + stencilLength / 2;
		for (int i = 0; i < constantTextLength; i++) {
			maximalLength += stringLength(constantText[i]);
		}
		for (int i = 0; i < stencilLength / 2; i++) {
			maximalLength +=
					(stringLength(stencil[i * 2])
							> stringLength(stencil[i * 2 + 1])) ?
							stringLength(stencil[i * 2]) :
							stringLength(stencil[i * 2 + 1]);
		}
		unsigned char combined[maximalLength];

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				sprintf((char*) combined, "%s %s %s %s %s", constantText[0],
						stencil[i], constantText[1], stencil[2 + j],
						constantText[2]);
				printf("%s\n", combined);

				unsigned char hash[4];
				reducedHash(combined, hash, 1);
				printHash(hash, 4);
				printf("\n");
			}
		}
	}

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

__global__ void birthdayAttack(unsigned char* hashs, unsigned int dim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int substringLength;
	if (x < dim) {
		sha256Context dummyContext;
		sha256Init(&dummyContext);
		for (int i = 0; i < 16; i++) {
			substringLength = goodTextOffsets[i + 1] - goodTextOffsets[i];
			sha256Update(&dummyContext, &goodText[goodTextOffsets[i]],
					substringLength);
			for (int j = 0; j < substringLength && x == 0; j++) {
				printf("%c", goodText[goodTextOffsets[i] + j]);
			}

			int number = x >> i & 0x00000001;
			substringLength = goodStencilOffsets[i * 2 + 1 + number]
					- goodStencilOffsets[i * 2 + number];
			sha256Update(&dummyContext,
					&goodStencil[goodStencilOffsets[i * 2 + number]],
					substringLength);
			for (int j = 0; j < substringLength && x == 0; j++) {
				printf("%c",
						goodStencil[goodStencilOffsets[i * 2 + number] + j]);
			}
		}
		substringLength = goodTextOffsets[17] - goodTextOffsets[16];
		sha256Update(&dummyContext, &goodText[goodTextOffsets[16]],
				substringLength);
		for (int j = 0; j < substringLength && x == 0; j++) {
			printf("%c", goodText[goodTextOffsets[16] + j]);
		}
		if (x == 0) {
			printf("\n");
		}

		unsigned char sha256hash[32];
		sha256Context context;
		sha256Init(&context);
		for (int j = 0; j < 1; j++) {
			if (x == 0) {
				unsigned char* text = (unsigned char*) "bla";
				sha256Update(&context, text, stringLength(text));
			} else {
				sha256Update(&context, goodText, stringLength(goodText));
			}
		}
		sha256Final(&context, sha256hash);

		reduceSha256(sha256hash, &hashs[x * 4]);
	}
	__syncthreads();

	if (x < dim) {
		unsigned char sha256hash[32];
		sha256Context context;
		sha256Init(&context);
		for (int j = 0; j < 1; j++) {
			sha256Update(&context, badText, stringLength(badText));
		}
		sha256Final(&context, sha256hash);

		unsigned char hash[4];
		reduceSha256(sha256hash, hash);

		for (unsigned int i = 0; i < dim; i++) {
			if (compareHash(&hashs[i * 4], hash, 4)) {
				//printf("Kollision! Gut: %d Böse: %d\n", i, x);
				collision = true;
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

__global__ void hashtestGPU() {
#ifndef NDEBUG
	testSha256LongInput();
	testReduceSha256();
#endif

	unsigned char hash[4];
	unsigned char* combined;
	combined = (unsigned char*) "This is a great day.";
	printf("%s\n", combined);
	reducedHash(combined, hash, 1);
	printHash(hash, 4);
	printf("\n");
	combined = (unsigned char*) "This is a bad day.";
	printf("%s\n", combined);
	reducedHash(combined, hash, 1);
	printHash(hash, 4);
	printf("\n");
	combined = (unsigned char*) "This was a great day.";
	printf("%s\n", combined);
	reducedHash(combined, hash, 1);
	printHash(hash, 4);
	printf("\n");
	combined = (unsigned char*) "This was a bad day.";
	printf("%s\n", combined);
	reducedHash(combined, hash, 1);
	printHash(hash, 4);
	printf("\n");
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
