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

extern std::map<std::string, Kernel> kernels;
extern std::map<std::string, int> kernelTPBs;
extern std::map<std::string, void(*)(Computation*)> kernelPrograms;
//extern std::map<std::string, void(*)(numb*, numb*, numb*)> kernelFDSs;
extern std::string selectedKernel;

extern std::map<std::string, Index> indices;

#define addKernel(name)         kernels[#name] = readKernelText(#name); \
                                kernelTPBs[#name] = THREADS_PER_BLOCK_##name; \
                                kernelPrograms[#name] = kernelProgram_##name;// \
                                //kernelFDSs[#name] = finiteDifferenceScheme_##name;
#define selectKernel(name)      selectedKernel = #name;
#define KERNEL      kernels[selectedKernel]
#define KERNEL_TPB  kernelTPBs[selectedKernel]
#define KERNEL_PROG kernelPrograms[selectedKernel]
//#define KERNEL_FDS  kernelFDSs[selectedKernel]
#define addIndex(name, fullname, function)  indices[name] = Index(fullname, AF_##function);

int compute(Computation*);

// Fill trajectory and parameter buffers with initial values
void fillAttributeBuffers(Computation* data, int* attributeStepIndices, unsigned long long startVariation, unsigned long long endVariation, bool onlyTrajectory);

// Set and copy map buffers for continuous map computation
void setMapValues(Computation*);