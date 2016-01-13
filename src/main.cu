//#define NDEBUG // include to remove asserts

// C standard header files
#include <stdio.h>

// CUDA header files
#include <cuda_runtime.h>

// own header files
#include "sha256.h"
#include "birthdayAttack.h"

__global__ void hashtestGPU();

int main(int argc, char* argv[]) {
#ifndef NDEBUG
	printf("DEBUG GPU:\n");
	hashtestGPU<<<1, 1>>>();
	cudaDeviceReset();
#endif

	birthdayAttack();
}

__global__ void hashtestGPU() {
	testSha256LongInput();
	testReduceSha256();
}
