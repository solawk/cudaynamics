#pragma once

#define V0(n) varValues[kernel::n]
#define V(n) data[stepStart + kernel::n]
#define P(n) paramValues[kernel::n]
#define M(n) maps[n]
#define NEXT kernel::VAR_COUNT

#define CUDA_SET_DEVICE cudaStatus = cudaSetDevice(0); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?"); goto Error; }

#define CUDA_MALLOC(address, size, failComment) cudaStatus = cudaMalloc(address, size); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, failComment); goto Error; }

#define CUDA_MEMCPY(dst, src, mode, size, failComment) cudaStatus = cudaMemcpy(dst, src, size, mode); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, failComment); goto Error; }

#define CUDA_LASTERROR cudaStatus = cudaGetLastError(); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

#define CUDA_SYNCHRONIZE cudaStatus = cudaDeviceSynchronize(); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus); goto Error; }

#define CUDA_RESET cudaStatus = cudaDeviceReset(); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceReset failed!\n"); hasFailed = true; }