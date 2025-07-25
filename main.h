#pragma once
#include <map>
#include <string>

#include "imgui_main.hpp"
#include "main_utils.h"
#include <objects.h>
#include "analysis.h"
#include "kernel_struct.h"
#include "computation_struct.h"
#include "mapData_struct.h"
#include "variationSteps.h"

#include "kernels/lorenz2/lorenz2.h"
#include "kernels/lorenzMod/lorenzMod.h"
#include "kernels/mrlcs_jj/mrlcs_jj.h"
#include "kernels/rlcs_jj/rlcs_jj.h"
#include "kernels/chen/chen.h"
#include "kernels/dadras/dadras.h"
#include "kernels/fourwing/fourwing.h"
#include "kernels/halvorsen/halvorsen.h"
#include "kernels/langford/langford.h"
#include "kernels/rossler/rossler.h"
#include "kernels/sprott/sprott.h"
#include "kernels/three_scroll/three_scroll.h"
#include "kernels/wilson/wilson.h"
#include "kernels/msprottj/msprottj.h"
#include "kernels/thomas/thomas.h"

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
void fillAttributeBuffers(Computation* data, int* attributeStepIndices, unsigned long long startVariation, unsigned long long endVariation, bool onlyTrajectory);

// Set and copy map buffers for continuous map computation
void setMapValues(Computation*);