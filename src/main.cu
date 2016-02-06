// C standard header files
#include <stdio.h>

// CUDA header files
#include <cuda_runtime.h>

// own header files
#include "sha256.h"
#include "birthdayAttack.h"
#include "helper.h"

__global__ void hashtestGPU();

int main(int argc, char* argv[]) {
#ifndef NDEBUG
	printf("DEBUG GPU:\n");
	hashtestGPU<<<1, 1>>>();
	cudaCheck(cudaDeviceReset());
#endif

	birthdayAttack();
	cudaCheck(cudaDeviceReset());
}

__global__ void hashtestGPU() {
	testSha256LongInput();
	testReduceSha256();
}
