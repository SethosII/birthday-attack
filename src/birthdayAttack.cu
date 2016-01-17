#include <math.h>
#include <stdio.h>
#include "sha256.h"
#include "birthdayAttack.h"

__constant__ unsigned char* badText =
		(unsigned char*) "Linux sucks! It is build  very old Software like X11 which makes them  to maintain. Also there are several projects within the Linux community which have mostly  like Wayland and Mir.  there is . This shows how  the community is. The next point are the . They suck. What should  stand for? Also the whole development  sucks. They have thousands of unpaid developers throwing code, part time, into a giant, internet-y  of software. What could possibly go wrong? The only result can be a  pile of . . As an old  goes: \"too many cooks spoil the broth\". So the freedom for the users consists of choices between lots of  projects. Even Linux users themself rant about it.  nobody should use Linux!";
__constant__ unsigned int badTextOffsets[] = { 0, 25, 70, 161, 184, 194, 211,
		253, 278, 317, 417, 486, 495, 497, 509, 615, 667, 692 };
__constant__ unsigned char* badStencil =
		(unsigned char*) "on top ofbased onharddifficultidentical aimthe same objectiveBecause of such thingsThereforea duplication of effortduplicate workdivideddisunitedcodenamescode namesUtopic UnicornTrusty TharprocessprocedurevattubgiantgiganticcrapgarbageThere is no focus in the developmentThe development lacks focussayingproverbhalf donesemi-finishedThat's whyTherefore";
__constant__ unsigned int badStencilOffsets[] = { 0, 9, 17, 21, 30, 43, 61, 83,
		92, 115, 129, 136, 145, 154, 164, 178, 189, 196, 205, 208, 211, 216,
		224, 228, 235, 271, 298, 304, 311, 320, 333, 343, 352 };

__constant__ unsigned char* goodText =
		(unsigned char*) "Linux is awesome! It is build  reliable Software like X11. Also there are several projects within the Linux community which have mostly  like Wayland and Mir - and that's . Why shouldn't you?  is no problem. And with  approaches you will propably find better solutions.  the many projects you can choose what  you best. The next point are the . They rule. How awesome was ? Also the whole development  is amazing. Some people argue that . They have thousands of unpaid developers throwing code, part time, into a giant, internet-y  of software. And it works. As an old  tells us: . Sometimes even Linux users themself rant about Linux - and they still love it because they can .  everybody should use Linux!";
__constant__ unsigned int goodTextOffsets[] = { 0, 30, 136, 171, 192, 217, 270,
		309, 343, 372, 401, 437, 531, 569, 580, 677, 679, 707 };
__constant__ unsigned char* goodStencil =
		(unsigned char*) "on top ofbased onidentical aimthe same objectivefinegreatDuplication of effortDuplicate workmoredifferentBecause ofThanks tosuitsfitscodenamescode namesHeisenbugBeefy Miracleprocessprocedurethere is no focus in the developmentthe development lacks focusvattubsayingproverb\"the more the merrier\"\"many hands make light work\"keep a critical eye on what they are usingcritically examineThat's whyTherefore";
__constant__ unsigned int goodStencilOffsets[] = { 0, 9, 17, 30, 48, 52, 57, 78,
		92, 96, 105, 115, 124, 129, 133, 142, 152, 161, 174, 181, 190, 226, 253,
		256, 259, 265, 272, 294, 322, 364, 382, 392, 401 };

__device__ int mutex = 0;

void birthdayAttack() {
	cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);

	unsigned int dim = pow(2, 16);

	unsigned char* hashs;
	cudaMalloc((void**) &hashs, dim * 4 * sizeof(unsigned char));

	int* collisions;
	cudaMalloc((void**) &collisions, 11 * sizeof(int));
	// initialize collisions or bad things will happen
	cudaMemset(collisions, 0, 11 * sizeof(int));

	dim3 blockDim(256);
	dim3 gridDim((dim + blockDim.x - 1) / blockDim.x);

	cudaEvent_t custart, custop;
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);
	cudaEventRecord(custart, 0);

	initBirthdayAttack<<<gridDim, blockDim>>>(hashs, dim);
	compareBirthdayAttack<<<gridDim, blockDim, blockDim.x * 4>>>(hashs, dim,
			collisions);

	cudaEventRecord(custop, 0);
	cudaEventSynchronize(custop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, custart, custop);
	printf("This birthday attack took %3.1f ms.\n", elapsedTime);
	cudaEventDestroy(custart);
	cudaEventDestroy(custop);

	printCollisions<<<1, 1>>>(collisions, hashs);

	cudaFree(hashs);
	cudaFree(collisions);
}

__global__ void compareBirthdayAttack(unsigned char* hashs, unsigned int dim,
		int* collisions) {
	int localThreadIndex = threadIdx.x;
	int blockSize = blockDim.x;
	int x = blockIdx.x * blockSize + localThreadIndex;
	int blockCount = dim / blockSize;
	if (x < dim) {
		const int reducedHashSize = 4;
		unsigned char hash[reducedHashSize];
		reduceHashFromStencil(x, hash, badText, badTextOffsets, badStencil,
				badStencilOffsets);

		int cacheSize = blockSize * reducedHashSize;
		extern __shared__ unsigned char cache[];
		for (int b = 0; b < blockCount; b++) {
#pragma unroll 4
			for (int i = 0; i < reducedHashSize; i++) {
				cache[localThreadIndex + i * blockSize] = hashs[localThreadIndex
						+ i * blockSize + b * cacheSize];
			}
			__syncthreads();

			for (unsigned int i = 0; i < blockSize; i++) {
				if (compareHash(&cache[i * reducedHashSize], hash, 4)) {
					lock();

					if (collisions[0] < 5) {
						// store good index
						collisions[collisions[0] * 2 + 1] = i
								+ b * cacheSize / 4;
						// store bad index
						collisions[(collisions[0] + 1) * 2] = x;
					}
					// increment number of collisions
					collisions[0]++;

					unlock();
				}
			}
			__syncthreads();
		}
	} else {
		// threads with nothing to do must also reach __synchthreads()
		for (int b = 0; b < blockCount; b++) {
			__syncthreads();
			__syncthreads();
		}
	}
}

__global__ void initBirthdayAttack(unsigned char* hashs, unsigned int dim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < dim) {
		reduceHashFromStencil(x, &hashs[x * 4], goodText, goodTextOffsets,
				goodStencil, goodStencilOffsets);
	}
}

__device__ void lock() {
	while (atomicCAS(&mutex, 0, 1) != 0) {
	}
}

__global__ void printCollisions(int* collisions, unsigned char* hashs) {
	if (collisions[0] > 0) {
		printf("Collisions found: %d\n\n", collisions[0]);
		for (int i = 0; i < collisions[0]; i++) {
			printf("\nCollision with good text #%d and bad text #%d\n",
					collisions[i * 2 + 1], collisions[(i + 1) * 2]);

			printf("\nGood plaintext:\n");
			printPlaintextOfIndex(collisions[i * 2 + 1], goodText,
					goodTextOffsets, goodStencil, goodStencilOffsets);
			printf("\nBad plaintext:\n");
			printPlaintextOfIndex(collisions[(i + 1) * 2], badText,
					badTextOffsets, badStencil, badStencilOffsets);

			printf("\nHash value of both is: ");
			printHash(&hashs[collisions[(i * 2 + 1)] * 4], 4);
			printf("\n");
		}
	} else {
		printf("No collisions found!\n");
	}
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

__device__ void reduceHashFromStencil(unsigned int x, unsigned char* hash,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets) {
	sha256Context context;
	sha256Init(&context);
	combineStencilForContext(x, &context, texts, textOffsets, stencils,
			stencilOffsets);
	unsigned char sha256hash[32];
	sha256Final(&context, sha256hash);
	reduceSha256(sha256hash, hash);
}

__device__ void unlock() {
	atomicExch(&mutex, 0);
}
