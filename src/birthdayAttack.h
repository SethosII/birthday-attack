#ifndef BIRTHDAYATTACK_H
#define BIRTHDAYATTACK_H

void birthdayAttack();
__device__ void combineStencilForContext(unsigned int x, sha256Context& context,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets);
__global__ void compareBirthdayAttack(unsigned char* hashs, unsigned int dim);
__global__ void initBirthdayAttack(unsigned char* hashs, unsigned int dim);
__device__ void printPlaintextOfIndex(unsigned int x, unsigned char* texts,
		unsigned int* textOffsets, unsigned char* stencils,
		unsigned int* stencilOffsets);
__device__ void reduceHashFromStencil(unsigned int x, unsigned char* hash,
		unsigned char* texts, unsigned int* textOffsets,
		unsigned char* stencils, unsigned int* stencilOffsets);

#endif
