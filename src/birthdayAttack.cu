#include <math.h>
#include <stdio.h>
#include "sha256.h"
#include "birthdayAttack.h"

__constant__ unsigned char* badText = (unsigned char*) ".................";
__constant__ unsigned int badTextOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17 };
__constant__ unsigned char* badStencil =
		(unsigned char*) "qqwweerrttzzuuiiooppaassddffgghhjjkkllyyxxccvvbbnnmm112233445566";
__constant__ unsigned int badStencilOffsets[] = { 0, 2, 4, 6, 8, 10, 12, 14, 16,
		18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
		54, 56, 58, 60, 62, 64 };

__constant__ unsigned char* goodText = (unsigned char*) ".................";
__constant__ unsigned int goodTextOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17 };
__constant__ unsigned char* goodStencil =
		(unsigned char*) "qwertzuiopasdfghjklyxcvbnm123456";
__constant__ unsigned int goodStencilOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
		28, 29, 30, 31, 32 };

__device__ int lock = 0;

__managed__ bool collision = false;

void birthdayAttack() {
	unsigned int dim = pow(2, 16);

	unsigned char* hashs;
	cudaMallocManaged((void**) &hashs, dim * 4 * sizeof(unsigned char));

	dim3 blockDim(256);
	dim3 gridDim((dim + blockDim.x - 1) / blockDim.x);

	cudaEvent_t custart, custop;
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);
	cudaEventRecord(custart, 0);

	initBirthdayAttack<<<gridDim, blockDim>>>(hashs, dim);
	compareBirthdayAttack<<<gridDim, blockDim, blockDim.x * 4>>>(hashs, dim);

	cudaEventRecord(custop, 0);
	cudaEventSynchronize(custop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, custart, custop);
	printf("This birthday attack took %3.1f ms.\n", elapsedTime);
	cudaEventDestroy(custart);
	cudaEventDestroy(custop);

	if (collision) {
		printf("Collisions found!");
	} else {
		printf("No collisions found!");
	}

	cudaFree(hashs);
}

__global__ void compareBirthdayAttack(unsigned char* hashs, unsigned int dim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < dim) {
		const int reducedHashSize = 4;
		unsigned char hash[reducedHashSize];
		reduceHashFromStencil(x, hash, badText, badTextOffsets, badStencil,
				badStencilOffsets);
		int blockSize = blockDim.x;
		int blockN = dim / blockSize;
		int cacheSize = blockSize * reducedHashSize;
		extern __shared__ unsigned char cache[];

		for (int b = 0; b < blockN; b++) {
			for (int i = 0; i < reducedHashSize; i++) {
				cache[threadIdx.x + i * blockSize] = hashs[threadIdx.x
						+ i * blockSize + b * cacheSize];
			}
			__syncthreads();

			for (unsigned int i = 0; i < blockSize; i++) {
				if (compareHash(&cache[i * reducedHashSize], hash,
						reducedHashSize)) {
					while (atomicCAS(&lock, 0, 1) != 0) {
					}
					printf("Collision, good plaintext:\n");
					printPlaintextOfIndex(
							threadIdx.x + i * blockSize + b * cacheSize,
							goodText, goodTextOffsets, goodStencil,
							goodStencilOffsets);
					printf("bad plaintext:\n");
					printPlaintextOfIndex(x, badText, badTextOffsets,
							badStencil, badStencilOffsets);
					printf("\n");
					collision = true;
					atomicExch(&lock, 0);
				}
			}
			__syncthreads();
		}
	} else {
		__syncthreads();
		__syncthreads();
	}
}

__device__ void combineStencilForContext(unsigned int x, sha256Context& context,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets) {
	unsigned int substringLength;
	for (int i = 0; i < 16; i++) {
		substringLength = textOffsets[i + 1] - textOffsets[i];
		sha256Update(&context, &texts[textOffsets[i]], substringLength);
		int number = x >> i & 0x00000001;
		substringLength = stencilOffsets[i * 2 + 1 + number]
				- stencilOffsets[i * 2 + number];
		sha256Update(&context, &stencils[stencilOffsets[i * 2 + number]],
				substringLength);
	}
	substringLength = textOffsets[17] - textOffsets[16];
	sha256Update(&context, &texts[textOffsets[16]], substringLength);
}

__device__ void printPlaintextOfIndex(unsigned int x, unsigned char* texts,
		unsigned int* textOffsets, unsigned char* stencils,
		unsigned int* stencilOffsets) {
	unsigned int substringLength;
	for (int i = 0; i < 16; i++) {
		substringLength = textOffsets[i + 1] - textOffsets[i];
		for (int j = 0; j < substringLength; j++) {
			printf("%c", texts[textOffsets[i] + j]);
		}
		int number = x >> i & 0x00000001;
		substringLength = stencilOffsets[i * 2 + 1 + number]
				- stencilOffsets[i * 2 + number];
		for (int j = 0; j < substringLength; j++) {
			printf("%c", stencils[stencilOffsets[i * 2 + number] + j]);
		}
	}
	substringLength = textOffsets[17] - textOffsets[16];
	for (int j = 0; j < substringLength; j++) {
		printf("%c", texts[textOffsets[16] + j]);
	}
	printf("\n");
}

__global__ void initBirthdayAttack(unsigned char* hashs, unsigned int dim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < dim) {
		reduceHashFromStencil(x, &hashs[x * 4], goodText, goodTextOffsets,
				goodStencil, goodStencilOffsets);
	}
}

__device__ void reduceHashFromStencil(unsigned int x, unsigned char* hash,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets) {
	sha256Context context;
	sha256Init(&context);
	combineStencilForContext(x, context, texts, textOffsets, stencils,
			stencilOffsets);
	unsigned char sha256hash[32];
	sha256Final(&context, sha256hash);
	reduceSha256(sha256hash, hash);
}
