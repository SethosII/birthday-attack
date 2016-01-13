#ifndef SHA256_H
#define SHA256_H
#include <stdbool.h>

typedef struct sha256Context {
	unsigned char data[64];
	unsigned int dataLength;
	unsigned int bitLength[2];
	unsigned int state[8];
} sha256Context;

__device__ unsigned int choice(unsigned int x, unsigned int y, unsigned int z);
__device__ bool compareHash(unsigned char hash1[], unsigned char hash2[],
		int length);
__device__ void doubleIntAdd(unsigned int* a, unsigned int* b, unsigned int c);
__device__ unsigned int epsilon0(unsigned int x);
__device__ unsigned int epsilon1(unsigned int x);
__device__ unsigned int majority(unsigned int x, unsigned int y,
		unsigned int z);
__device__ void printHash(unsigned char hash[], int length);
__device__ void reducedHash(unsigned char data[], unsigned char hash[],
		unsigned int iterations);
__device__ void reduceSha256(unsigned char hash[], unsigned char reducedHash[]);
__device__ unsigned int rotateRight(unsigned int a, unsigned int b);
__device__ void sha256(unsigned char data[], unsigned char hash[],
		unsigned int iterations);
__device__ void sha256Init(sha256Context *context);
__device__ void sha256Final(sha256Context *context, unsigned char hash[]);
__device__ void sha256Transform(sha256Context *context, unsigned char data[]);
__device__ void sha256Update(sha256Context *context, unsigned char data[],
		unsigned int length);
__device__ unsigned int sigma0(unsigned int x);
__device__ unsigned int sigma1(unsigned int x);
__device__ unsigned int stringLength(unsigned char* str);
__device__ void testReduceSha256();
__device__ void testSha256LongInput();

#endif
