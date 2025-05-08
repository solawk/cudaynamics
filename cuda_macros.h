#pragma once

// Current variable value in the FDS
#define V(n)			currentV[attributes::variables::n]

// Next variable value in the FDS
#define Vnext(n)		nextV[attributes::variables::n]

// Parameter value in the FDS
#define P(n)			parameters[attributes::parameters::n]

// Map
#define M(n)			CUDA_kernel.mapDatas[attributes::maps::n]

// Computation access macros

#define CUDA_marshal	data->marshal
#define CUDA_kernel		CUDA_marshal.kernel

#define STEP_INDICES_X(m) CUDA_marshal.stepIndices[indicesStart + (CUDA_kernel.mapDatas[attributes::maps::m].typeX == PARAMETER ? CUDA_kernel.VAR_COUNT : 0) + CUDA_kernel.mapDatas[attributes::maps::m].indexX]
#define STEP_INDICES_Y(m) CUDA_marshal.stepIndices[indicesStart + (CUDA_kernel.mapDatas[attributes::maps::m].typeY == PARAMETER ? CUDA_kernel.VAR_COUNT : 0) + CUDA_kernel.mapDatas[attributes::maps::m].indexY]

// Preparation macros

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

// Computation macros

#define NORM_3D(x1, x2, y1, y2, z1, z2) sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1))

// == Largest Lyapunov Exponent (LLE) ==

// Store to temp array
#define LLE_STORE_TO_TEMP for (int i = 0; i < kernel::VAR_COUNT; i++) LLE_array_temp[i] = LLE_array[i];

// Parallel array access
#define LLE_V(n) LLE_array[kernel::n]

// Temp parallel array access
#define LLE_V_PREV(n) LLE_array_temp[kernel::n]

// Initialize parallel array
#define LLE_FILL for (int i = 0; i < kernel::VAR_COUNT; i++) LLE_array[i] = data[stepStart + i];

// Add value "by" to variable "var" in parallel array
#define LLE_DEFLECT(var, by) LLE_array[kernel::var] += by;

// Retract variable "var" by a divider "div"
#define LLE_RETRACT(var, div) LLE_array[kernel::var] = data[stepStart + kernel::var NEXT] + (LLE_array[kernel::var] - data[stepStart + kernel::var NEXT]) / div;

// If-condition for every n-th step "s"
#define LLE_IF_MOD(s, nth) if (s % nth == 0)

// Add calculated local LLE
#define LLE_ADD(value) LLE_div++; LLE_value += value;

// Mean LLE
#define LLE_MEAN_RESULT LLE_value / LLE_div