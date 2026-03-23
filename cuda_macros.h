#pragma once

// Current variable value in the FDS
#define V(n)			currentV[attributes::variables::n]

// Next variable value in the FDS
#define Vnext(n)		nextV[attributes::variables::n]

// Parameter value in the FDS
#define P(n)			parameters[attributes::parameters::n]

//#define H_BRANCH(p, v)  (CUDA_kernel.stepType == 0 ? p : (CUDA_kernel.stepType == 1 ? v : (numb)1.0))
#define H_BRANCH(p, v)  p

#define H               parameters[attributes::parameters::COUNT]

#define H_KERNEL        parameters[CUDA_kernel.PARAM_COUNT - 1]

#define LOCAL_BUFFERS   numb variables[MAX_ATTRIBUTES]{0}; \
                        numb variablesNext[MAX_ATTRIBUTES]{0}; \
                        numb parameters[MAX_ATTRIBUTES]{0};

#define LOAD_ATTRIBUTES(isAnalysis)     for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) variables[i] = CUDA_marshal.variableInits[variation * CUDA_kernel.VAR_COUNT + i]; \
                                        for (int i = 0; i < CUDA_kernel.PARAM_COUNT; i++) parameters[i] = CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT + i]; \
                                        if (!isAnalysis && !data->isHires) for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) CUDA_marshal.trajectory[variationStart + i] = variables[i];

// Map
#define M(n)			CUDA_kernel.mapDatas[attributes::maps::n]

// Method
#define ifMETHOD(p, n)  if ((int)p == attributes::methods::n) 

// Signal
#define ifSIGNAL(p, n)  if ((int)p == attributes::waveforms::n)

// Map settings
#define MS(map, offset) 0.0

#define MO(map)         (CUDA_kernel.mapDatas[attributes::maps::map].offset * CUDA_marshal.totalVariations)

#define mapPosition     (variation + offset)
#define mapValueAt(index) (variation + offset) + (index * CUDA_marshal.totalVariations)

#define indexPosition(offset, value)  ((offset + value) * CUDA_marshal.totalVariations + variation + (data->isGPU ? 0 : data->startVariationInCurrentExecute))

// Computation access macros

#define CUDA_marshal	data->marshal
#define CUDA_kernel		CUDA_marshal.kernel
#define CUDA_ATTR_COUNT (CUDA_kernel.VAR_COUNT + CUDA_kernel.PARAM_COUNT)

//#define STEP_INDICES_X(m) CUDA_marshal.stepIndices[indicesStart + (CUDA_kernel.mapDatas[attributes::maps::m].typeX == PARAMETER ? CUDA_kernel.VAR_COUNT : 0) + CUDA_kernel.mapDatas[attributes::maps::m].indexX]
//#define STEP_INDICES_Y(m) CUDA_marshal.stepIndices[indicesStart + (CUDA_kernel.mapDatas[attributes::maps::m].typeY == PARAMETER ? CUDA_kernel.VAR_COUNT : 0) + CUDA_kernel.mapDatas[attributes::maps::m].indexY]

// Preparation macros

#define CUDA_SET_DEVICE cudaStatus = cudaSetDevice(0); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n"); goto Error; }

#define CUDA_MALLOC(address, size, failComment) cudaStatus = cudaMalloc(address, size); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, failComment); goto Error; }

#define CUDA_MEMCPY(dst, src, mode, size, failComment) cudaStatus = cudaMemcpy(dst, src, size, mode); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, failComment); goto Error; }

#define CUDA_LASTERROR cudaStatus = cudaGetLastError(); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

#define CUDA_SYNCHRONIZE cudaStatus = cudaDeviceSynchronize(); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus); goto Error; }

#define CUDA_RESET cudaStatus = cudaDeviceReset(); \
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceReset failed!\n");/* hasFailed = true;*/ }

// Computation macros

#define FDS_ARGUMENTS   &(variables[0]), &(variablesNext[0]), &(parameters[0])

// Skip transient steps using the provided finite difference scheme, akin to computing but without recording the trajectory
#define DEC_TRANSIENT_STEPS (!CUDA_kernel.usingTime ? CUDA_kernel.transientSteps : CUDA_kernel.transientTime / H_KERNEL)

#define DEC_STEPS       (!CUDA_kernel.usingTime ? CUDA_kernel.steps : CUDA_kernel.time / H_KERNEL)

#define TRANSIENT_SKIP_NEW(FDS)      if (DEC_TRANSIENT_STEPS > 0 && data->isFirst)  \
                            {  \
                                numb transientBuffer[MAX_ATTRIBUTES];  \
                                for (int ts = 0; ts < DEC_TRANSIENT_STEPS; ts++)  \
                                {  \
                                    FDS(FDS_ARGUMENTS);  \
                                    for (int v = 0; v < CUDA_kernel.VAR_COUNT; v++) variables[v] = variablesNext[v];  \
                                }  \
                                if (!data->isHires) for (int v = 0; v < CUDA_kernel.VAR_COUNT; v++) CUDA_marshal.trajectory[variationStart + v] = variables[v]; \
                                else for (int v = 0; v < CUDA_kernel.VAR_COUNT; v++) CUDA_marshal.variableInits[variation * CUDA_kernel.VAR_COUNT + v] = variables[v]; \
                            }

#define FDS_ARGUMENTS   &(variables[0]), &(variablesNext[0]), &(parameters[0])

#define TRANSFER_VARIABLES  for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) variables[i] = variablesNext[i];

#define RECORD_STEP     if (!data->isHires) for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i] = variables[i]; \
                        else if (s == DEC_STEPS - 1) for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) CUDA_marshal.variableInits[variation * CUDA_kernel.VAR_COUNT + i] = variables[i];

#define TARGET_STEPS    CUDA_kernel.targetSteps

#define NORMAL_STEP_IN_ANALYSIS_IF_HIRES    if (data->isHires) { finiteDifferenceScheme(FDS_ARGUMENTS); TRANSFER_VARIABLES; RECORD_STEP; }

//#define RECORD_STEP     for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) CUDA_marshal.trajectory[stepStart + CUDA_kernel.VAR_COUNT + i] = variables[i] = variablesNext[i];

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