#pragma once

// Current variable value in the FDS
#define V(n)			currentV[attributes::variables::n]

// Next variable value in the FDS
#define Vnext(n)		nextV[attributes::variables::n]

// Parameter value in the FDS
#define P(n)			parameters[attributes::parameters::n]

//#define H_BRANCH(p, v)  (CUDA_kernel.stepType == 0 ? p : (CUDA_kernel.stepType == 1 ? v : (numb)1.0))
#define H_BRANCH(p, v)  p

//#define H               H_BRANCH(parameters[CUDA_kernel.PARAM_COUNT - 1], currentV[CUDA_kernel.VAR_COUNT - 1])
#define H               parameters[attributes::parameters::COUNT]

#define LOCAL_BUFFERS   numb variables[MAX_ATTRIBUTES]{0}; \
                        numb variablesNext[MAX_ATTRIBUTES]{0}; \
                        numb parameters[MAX_ATTRIBUTES]{0}; \
                        PerThread pt;

#define LOAD_ATTRIBUTES(isAnalysis)     for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) variables[i] = CUDA_marshal.variableInits[variation * CUDA_kernel.VAR_COUNT + i]; \
                                        for (int i = 0; i < CUDA_kernel.PARAM_COUNT; i++) parameters[i] = CUDA_marshal.parameterVariations[variation * CUDA_kernel.PARAM_COUNT + i]; \
                                        if (!isAnalysis && !data->isHires) for (int i = 0; i < CUDA_kernel.VAR_COUNT; i++) CUDA_marshal.trajectory[variationStart + i] = variables[i];

#define LOAD_PT_OMP     pt.randomCPUgen = data->randomCPUgen[omp_get_thread_num()]; \
                        pt.randomCPUdistrib = data->randomCPUdistrib[omp_get_thread_num()];

#define LOAD_PT_CUDA    curandState state; \
                        curand_init(data->randomSeed, 0L, 0L, &state); \
                        pt.randomGPUstate = &state;

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