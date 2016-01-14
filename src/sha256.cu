#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "sha256.h"

__constant__ unsigned int C[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf,
		0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
		0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
		0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
		0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
		0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
		0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e,
		0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
		0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c,
		0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee,
		0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
		0xc67178f2 };

__device__ void printHash(unsigned char* hash, int length) {
	for (int i = 0; i < length; i++) {
		printf("%02x", hash[i]);
	}
	printf("\n");
}

__device__ void reduceSha256(unsigned char* sha256hash,
		unsigned char* reducedHash) {
	for (int i = 0; i < 4; i++) {
		reducedHash[i] = sha256hash[i];
	}
	for (int j = 1; j < 8; j++) {
		for (int i = 0; i < 4; i++) {
			reducedHash[i] ^= sha256hash[j * 4 + i];
		}
	}
}

__device__ void reducedHash(unsigned char* data, unsigned char* hash,
		unsigned int iterations) {
	unsigned char sha256hash[32];
	sha256(data, sha256hash, iterations);
	reduceSha256(sha256hash, hash);
}

__device__ void sha256Init(sha256Context* context) {
	context->dataLength = 0;
	context->bitLength[0] = 0;
	context->bitLength[1] = 0;
	context->state[0] = 0x6a09e667;
	context->state[1] = 0xbb67ae85;
	context->state[2] = 0x3c6ef372;
	context->state[3] = 0xa54ff53a;
	context->state[4] = 0x510e527f;
	context->state[5] = 0x9b05688c;
	context->state[6] = 0x1f83d9ab;
	context->state[7] = 0x5be0cd19;
}

__device__ void sha256(unsigned char* data, unsigned char* hash,
		unsigned int iterations) {
	sha256Context context;

	sha256Init(&context);
	for (int j = 0; j < iterations; j++) {
		sha256Update(&context, data, stringLength(data));
	}
	sha256Final(&context, hash);
}

__device__ void sha256Final(sha256Context* context, unsigned char* hash) {
	unsigned int length = context->dataLength;

	// pad data in the buffer.
	if (context->dataLength < 56) {
		context->data[length++] = 0x80;
		for (; length < 56; length++) {
			context->data[length] = 0x00;
		}
	} else {
		context->data[length++] = 0x80;
		for (; length < 64; length++) {
			context->data[length] = 0x00;
		}
		sha256Transform(context, context->data);
		memset(context->data, 0, 56);
	}

	// append the total message length in bits and transform.
	doubleIntAdd(&context->bitLength[0], &context->bitLength[1],
			context->dataLength * 8);
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 4; i++) {
			context->data[63 - i - 4 * j] = context->bitLength[j] >> 8 * i;
		}
	}
	sha256Transform(context, context->data);

	// implementation uses little endian byte ordering and SHA uses big endian, reverse all bytes
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 8; j++) {
			hash[i + 4 * j] = (context->state[j] >> (24 - i * 8)) & 0x000000ff;
		}
	}
}

__device__ void sha256Transform(sha256Context* context, unsigned char* data) {
	unsigned int shadowRegister[8];
	unsigned int messageSchedule[64];

	for (int i = 0, j = 0; i < 16; i++, j += 4) {
		messageSchedule[i] = (data[j] << 24) | (data[j + 1] << 16)
				| (data[j + 2] << 8) | (data[j + 3]);
	}
	for (int i = 16; i < 64; i++) {
		messageSchedule[i] = sigma1(messageSchedule[i - 2])
				+ messageSchedule[i - 7] + sigma0(messageSchedule[i - 15])
				+ messageSchedule[i - 16];
	}

	for (int i = 0; i < 8; i++) {
		shadowRegister[i] = context->state[i];
	}

	for (int i = 0; i < 64; i++) {
		unsigned int textRegister1 = shadowRegister[7]
				+ epsilon1(shadowRegister[4])
				+ choice(shadowRegister[4], shadowRegister[5],
						shadowRegister[6]) + C[i] + messageSchedule[i];
		unsigned int textRegister2 = epsilon0(shadowRegister[0])
				+ majority(shadowRegister[0], shadowRegister[1],
						shadowRegister[2]);
		for (int j = 7; j > 0; j--) {
			shadowRegister[j] = shadowRegister[j - 1];
		}
		shadowRegister[0] = textRegister1 + textRegister2;
		shadowRegister[4] += textRegister1;
	}

	for (int i = 0; i < 8; i++) {
		context->state[i] += shadowRegister[i];
	}
}

__device__ void sha256Update(sha256Context* context, unsigned char* data,
		unsigned int length) {
	for (unsigned int i = 0; i < length; i++) {
		context->data[context->dataLength] = data[i];
		context->dataLength++;
		if (context->dataLength == 64) {
			sha256Transform(context, context->data);
			doubleIntAdd(&context->bitLength[0], &context->bitLength[1], 512);
			context->dataLength = 0;
		}
	}
}

/*
 * This is a great day.
 * 93784ef2 2ab7c997 d0026ee8 8c2b37d0 5cdfd410 d41b84fb 90653697 e471d6e6
 *
 * 1001 0011 0111 1000 0100 1110 1111 0010
 * 0010 1010 1011 0111 1100 1001 1001 0111
 * 1101 0000 0000 0010 0110 1110 1110 1000
 * 1000 1100 0010 1011 0011 0111 1101 0000
 * 0101 1100 1101 1111 1101 0100 0001 0000
 * 1101 0100 0001 1011 1000 0100 1111 1011
 * 1001 0000 0110 0101 0011 0110 1001 0111
 * 1110 0100 0111 0001 1101 0110 1110 0110
 * ---------------------------------------
 * 0001 1001 0011 0110 0110 1110 1100 0111
 *
 * 19366ec7
 */
__device__ void testReduceSha256() {
	// text: unsigned char combined[] = "This is a great day.";
	unsigned char specifiedHash[] = { 0x19, 0x36, 0x6e, 0xc7 };
	unsigned char sha256hash[] = { 0x93, 0x78, 0x4e, 0xf2, 0x2a, 0xb7, 0xc9,
			0x97, 0xd0, 0x02, 0x6e, 0xe8, 0x8c, 0x2b, 0x37, 0xd0, 0x5c, 0xdf,
			0xd4, 0x10, 0xd4, 0x1b, 0x84, 0xfb, 0x90, 0x65, 0x36, 0x97, 0xe4,
			0x71, 0xd6, 0xe6 };
	unsigned char hash[4];

	reduceSha256(sha256hash, hash);

	assert(compareHash(specifiedHash, hash, 4));
	printf("DEBUG: testReduceSha256 passed\n");
}

__device__ void testSha256LongInput() {
	unsigned char longText[] = "aaaaaaaaaa";
	unsigned char specifiedHash[] = { 0xcd, 0xc7, 0x6e, 0x5c, 0x99, 0x14, 0xfb,
			0x92, 0x81, 0xa1, 0xc7, 0xe2, 0x84, 0xd7, 0x3e, 0x67, 0xf1, 0x80,
			0x9a, 0x48, 0xa4, 0x97, 0x20, 0x0e, 0x04, 0x6d, 0x39, 0xcc, 0xc7,
			0x11, 0x2c, 0xd0 };
	unsigned char hash[32];

	sha256(longText, hash, 100000);

	assert(compareHash(specifiedHash, hash, 32));
	printf("DEBUG: testSha256LongInput passed\n");
}
