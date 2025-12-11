#pragma once
#include <map>
#include <string>

#include "imgui_main.hpp"
#include "main_utils.h"
#include <objects.h>
#include "kernel_struct.h"
#include "computation_struct.h"
#include "mapData_struct.h"
#include "variationSteps.h"
#include "systemsHeaders.h"
#include "abstractSettings_struct.h"
#include "kernel_map.h"
#include "indices_map.h"

//extern std::map<AnalysisIndex, Index> indices;

#define addKernel(name)         kernels[#name] = readKernelText(#name); \
                                kernelPrograms[#name] = kernelProgram_##name; \
                                kernelWrappers[#name] = gpu_wrapper_##name;
#define selectKernel(name)      selectedKernel = #name;
#define KERNEL      kernels[selectedKernel]
#define KERNEL_PROG kernelPrograms[selectedKernel]
#define KERNEL_GPU  kernelWrappers[selectedKernel]

#define addIndex(name, fullname, function, size)  indices[name] = Index(fullname, ANF_##function, size);

int compute(Computation*);

// Fill trajectory and parameter buffers with initial values
void fillAttributeBuffers(Computation* data, int* attributeStepIndices, unsigned long long startVariation, unsigned long long endVariation, bool onlyTrajectory);

void setupAnFuncs(Computation*);

void indexDecayPostprocessing(Computation*);