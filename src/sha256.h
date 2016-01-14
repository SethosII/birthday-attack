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
__device__ bool compareHash(unsigned char* hash1, unsigned char* hash2,
		int length);
__device__ void doubleIntAdd(unsigned int* a, unsigned int* b, unsigned int c);
__device__ unsigned int epsilon0(unsigned int x);
__device__ unsigned int epsilon1(unsigned int x);
__device__ unsigned int majority(unsigned int x, unsigned int y,
		unsigned int z);
__device__ void printHash(unsigned char* hash, int length);
__device__ void reducedHash(unsigned char* data, unsigned char* hash,
		unsigned int iterations);
__device__ void reduceSha256(unsigned char* hash, unsigned char* reducedHash);
__device__ unsigned int rotateRight(unsigned int a, unsigned int b);
__device__ void sha256(unsigned char* data, unsigned char* hash,
		unsigned int iterations);
__device__ void sha256Init(sha256Context* context);
__device__ void sha256Final(sha256Context* context, unsigned char* hash);
__device__ void sha256Transform(sha256Context* context, unsigned char* data);
__device__ void sha256Update(sha256Context* context, unsigned char* data,
		unsigned int length);
__device__ unsigned int sigma0(unsigned int x);
__device__ unsigned int sigma1(unsigned int x);
__device__ unsigned int stringLength(unsigned char* str);
__device__ void testReduceSha256();
__device__ void testSha256LongInput();

__forceinline__ __device__ unsigned int choice(unsigned int x, unsigned int y,
		unsigned int z) {
	return (x & y) ^ (~x & z);
}

__forceinline__ __device__ bool compareHash(unsigned char* hash1,
		unsigned char* hash2, int length) {
	bool equal = true;
#pragma unroll 4
	for (int i = 0; i < length; i++) {
		equal = equal && hash1[i] == hash2[i];
	}
	return equal;
}

// doubleIntAdd treats two unsigned ints a and b as one 64-bit integer and adds c to it
__forceinline__ __device__ void doubleIntAdd(unsigned int* a, unsigned int* b,
		unsigned int c) {
	if (*a > (0xffffffff - c)) {
		*b++;
	}
	*a += c;
}

__forceinline__ __device__ unsigned int epsilon0(unsigned int x) {
	return rotateRight(x, 2) ^ rotateRight(x, 13) ^ rotateRight(x, 22);
}

__forceinline__ __device__ unsigned int epsilon1(unsigned int x) {
	return rotateRight(x, 6) ^ rotateRight(x, 11) ^ rotateRight(x, 25);
}

__forceinline__ __device__ unsigned int majority(unsigned int x, unsigned int y,
		unsigned int z) {
	return (x & y) ^ (x & z) ^ (y & z);
}

__forceinline__ __device__ unsigned int rotateRight(unsigned int a,
		unsigned int b) {
	return (a >> b) | (a << (32 - b));
}

__forceinline__ __device__ unsigned int sigma0(unsigned int x) {
	return rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >> 3);
}

__forceinline__ __device__ unsigned int sigma1(unsigned int x) {
	return rotateRight(x, 17) ^ rotateRight(x, 19) ^ (x >> 10);
}

__forceinline__ __device__ unsigned int stringLength(unsigned char* str) {
	unsigned int length;
	for (length = 0; str[length] != '\0'; length++) {
	}
	return length;
}

#endif
