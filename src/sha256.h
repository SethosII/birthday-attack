#ifndef SHA256_H
#define SHA256_H
#include <stdbool.h>

typedef struct sha256Context {
	unsigned char data[64];
	unsigned int dataLength;
	unsigned int bitLength[2];
	unsigned int state[8];
} sha256Context;

unsigned int choice(unsigned int x, unsigned int y, unsigned int z);
bool compareHash(unsigned char hash1[], unsigned char hash2[], int length);
void copyHash(unsigned char from[], unsigned char to[], int length);
void doubleIntAdd(unsigned int* a, unsigned int* b, unsigned int c);
unsigned int epsilon0(unsigned int x);
unsigned int epsilon1(unsigned int x);
unsigned int majority(unsigned int x, unsigned int y, unsigned int z);
void printHash(unsigned char hash[], int length);
void reducedHash(unsigned char data[], unsigned char hash[], unsigned int iterations);
void reduceSha256(unsigned char hash[], unsigned char reducedHash[]);
unsigned int rotateRight(unsigned int a, unsigned int b);
void sha256(unsigned char data[], unsigned char hash[], unsigned int iterations);
void sha256Init(sha256Context *context);
void sha256Final(sha256Context *context, unsigned char hash[]);
void sha256Transform(sha256Context *context, unsigned char data[]);
void sha256Update(sha256Context *context, unsigned char data[], unsigned int length);
unsigned int sigma0(unsigned int x);
unsigned int sigma1(unsigned int x);
void testReduceSha256();
void testSha256LongInput();

#endif
