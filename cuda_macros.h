#pragma once

// Short kernel macros

// Variable value in the current step and variation
#define dataV(n)		data[stepStart + kernel::n]

// Starting value of the variable in the variation
#define initV(n)		dataV(n) = varValues[kernel::n]

// Current variable value in the FDS
#define V(n)			currentV[kernel::n]

// Next variable value in the FDS
#define Vnext(n)		nextV[kernel::n]

// Parameter value in the FDS
#define P(n)			parameters[kernel::n]

// Map cell value (index, x, y)
#define M(i, x, y)		maps[mapData[i].offset + y * mapData[i].xSize + x]

// Shift to next step
#define NEXT			+ kernel::VAR_COUNT

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

// Declare parallel array
#define LLE_INIT(type) type LLE_array[kernel::VAR_COUNT]; type LLE_array_temp[kernel::VAR_COUNT]; float LLE_value = 0.0f; int LLE_div = 0;

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