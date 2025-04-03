#pragma once
#include <map>
#include <string>

#include "imgui_main.h"
#include "main_utils.h"
#include <objects.h>
#include "analysis.h"
#include "kernel_struct.h"
#include "computation_struct.h"
#include "mapData_struct.h"
#include "variationSteps.h"

#include "kernels/lorenz2/lorenz2.h"
#include "kernels/halvorsen/halvorsen.h"
#include "kernels/mrlcs_jj/mrlcs_jj.h"

extern std::map<std::string, Kernel> kernels;
extern std::map<std::string, int> kernelTPBs;
extern std::map<std::string, void(*)(Computation*)> kernelPrograms;
extern std::map<std::string, void(*)(numb*, numb*, numb*, numb)> kernelFDSs;
extern std::string selectedKernel;

#define addKernel(name)         kernels[#name] = readKernelText(#name); \
                                kernelTPBs[#name] = THREADS_PER_BLOCK_##name; \
                                kernelPrograms[#name] = kernelProgram_##name; \
                                kernelFDSs[#name] = finiteDifferenceScheme_##name;
#define selectKernel(name)      selectedKernel = #name;
#define KERNEL      kernels[selectedKernel]
#define KERNEL_TPB  kernelTPBs[selectedKernel]
#define KERNEL_PROG kernelPrograms[selectedKernel]
#define KERNEL_FDS  kernelFDSs[selectedKernel]

int compute(Computation*);

// Fill trajectory and parameter buffers with initial values
void fillAttributeBuffers(Computation*);