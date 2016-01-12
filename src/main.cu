//#define NDEBUG // include to remove asserts

// C standard header files
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA header files
#include <cuda_runtime.h>

// own header files
#include "sha256.h"
#include "birthdayAttack.h"

void hashtestCPU();
__global__ void hashtestGPU();

int main(int argc, char* argv[]) {
#ifndef NDEBUG
	printf("DEBUG GPU:\n");
	hashtestGPU<<<1, 1>>>();
	cudaDeviceReset();
	printf("DEBUG CPU:\n");
	hashtestCPU();
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
