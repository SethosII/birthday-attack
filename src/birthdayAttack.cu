#include <stdio.h>
#include "sha256.h"
#include "birthdayAttack.h"

__constant__ unsigned char* goodText = (unsigned char*) ".................";
__constant__ unsigned int goodTextOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17 };
__constant__ unsigned char* goodStencil =
		(unsigned char*) "qwertzuiopasdfghjklyxcvbnm123456";
__constant__ unsigned int goodStencilOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
		28, 29, 30, 31, 32 };

__constant__ unsigned char* badText = (unsigned char*) ".................";
__constant__ unsigned int badTextOffsets[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17 };
__constant__ unsigned char* badStencil =
		(unsigned char*) "qqwweerrttzzuuiiooppaassddffgghhjjkkllyyxxccvvbbnnmm112233445566";
__constant__ unsigned int badStencilOffsets[] = { 0, 2, 4, 6, 8, 10, 12, 14, 16,
		18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
		54, 56, 58, 60, 62, 64 };

__managed__ bool collision = false;

__global__ void birthdayAttack(unsigned char* hashs, unsigned int dim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int substringLength;
	if (x < dim) {
		sha256Context dummyContext;
		sha256Init(&dummyContext);
		for (int i = 0; i < 16; i++) {
			substringLength = goodTextOffsets[i + 1] - goodTextOffsets[i];
			sha256Update(&dummyContext, &goodText[goodTextOffsets[i]],
					substringLength);
			for (int j = 0; j < substringLength && x == 0; j++) {
				printf("%c", goodText[goodTextOffsets[i] + j]);
			}

			int number = x >> i & 0x00000001;
			substringLength = goodStencilOffsets[i * 2 + 1 + number]
					- goodStencilOffsets[i * 2 + number];
			sha256Update(&dummyContext,
					&goodStencil[goodStencilOffsets[i * 2 + number]],
					substringLength);
			for (int j = 0; j < substringLength && x == 0; j++) {
				printf("%c",
						goodStencil[goodStencilOffsets[i * 2 + number] + j]);
			}
		}
		substringLength = goodTextOffsets[17] - goodTextOffsets[16];
		sha256Update(&dummyContext, &goodText[goodTextOffsets[16]],
				substringLength);
		for (int j = 0; j < substringLength && x == 0; j++) {
			printf("%c", goodText[goodTextOffsets[16] + j]);
		}
		if (x == 0) {
			printf("\n");
		}

		unsigned char sha256hash[32];
		sha256Context context;
		sha256Init(&context);
		for (int j = 0; j < 1; j++) {
			if (x == 0) {
				unsigned char* text = (unsigned char*) "bla";
				sha256Update(&context, text, stringLength(text));
			} else {
				sha256Update(&context, goodText, stringLength(goodText));
			}
		}
		sha256Final(&context, sha256hash);

		reduceSha256(sha256hash, &hashs[x * 4]);
	}
	__syncthreads();

	if (x < dim) {
		unsigned char sha256hash[32];
		sha256Context context;
		sha256Init(&context);
		for (int j = 0; j < 1; j++) {
			sha256Update(&context, badText, stringLength(badText));
		}
		sha256Final(&context, sha256hash);

		unsigned char hash[4];
		reduceSha256(sha256hash, hash);

		for (unsigned int i = 0; i < dim; i++) {
			if (compareHash(&hashs[i * 4], hash, 4)) {
				//printf("Kollision! Gut: %d Böse: %d\n", i, x);
				collision = true;
			}
		}
	}
}

