#ifndef BIRTHDAYATTACK_H
#define BIRTHDAYATTACK_H

void birthdayAttack();
__device__ void combineStencilForContext(unsigned int x, sha256Context* context,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets);
__global__ void compareBirthdayAttack(unsigned char* hashs, unsigned int dim,
		int* collision);
__global__ void initBirthdayAttack(unsigned char* hashs, unsigned int dim);
__device__ void lock();
__global__ void printCollisions(int* collisions, unsigned char* hashs);
__device__ void printPlaintextOfIndex(unsigned int x, unsigned char* texts,
		unsigned int* textOffsets, unsigned char* stencils,
		unsigned int* stencilOffsets);
__device__ void reduceHashFromStencil(unsigned int x, unsigned char* hash,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets);
__device__ void unlock();

__forceinline__ __device__ void combineStencilForContext(unsigned int x,
		sha256Context* context, unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets) {
	unsigned int substringLength;
	for (int i = 0; i < 16; i++) {
		substringLength = textOffsets[i + 1] - textOffsets[i];
		sha256Update(context, &texts[textOffsets[i]], substringLength);
		int number = x >> i & 0x00000001;
		substringLength = stencilOffsets[i * 2 + 1 + number]
				- stencilOffsets[i * 2 + number];
		sha256Update(context, &stencils[stencilOffsets[i * 2 + number]],
				substringLength);
	}
	substringLength = textOffsets[17] - textOffsets[16];
	sha256Update(context, &texts[textOffsets[16]], substringLength);
}

#endif
