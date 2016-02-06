#ifndef HELPER_H
#define HELPER_H
//#define NDEBUG // include to remove asserts and cudaCheck
#define cudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)

inline void __cudaCheck(cudaError err, const char* file, int line) {
#ifndef NDEBUG
	if (err != cudaSuccess) {
		fprintf(stderr, "%s(%d): CUDA error: %s\n", __FILE__, __LINE__,
				cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
#endif
}

#endif
