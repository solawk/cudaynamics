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


#include "systems/lorenz/lorenz.h"
#include "systems/lorenzVar/lorenzVar.h"
#include "systems/lorenz83/lorenz83.h"
#include "systems/hindmarsh_rose/hindmarsh_rose.h"
#include "systems/izhikevich/izhikevich.h"
#include "systems/jj_mrlcs/jj_mrlcs.h"
#include "systems/jj_rlcs/jj_rlcs.h"
#include "systems/chen/chen.h"
#include "systems/dadras/dadras.h"
#include "systems/fourwing/fourwing.h"
#include "systems/halvorsen/halvorsen.h"
#include "systems/langford/langford.h"
#include "systems/rossler/rossler.h"
#include "systems/sprott14/sprott14.h"
#include "systems/sprottJm/sprottJm.h"
#include "systems/three_scroll/three_scroll.h"
#include "systems/wilson/wilson.h"
#include "systems/sang/sang.h"
#include "systems/rabinovich_fabrikant/rabinovich_fabrikant.h"
#include "systems/thomas/thomas.h"
#include "systems/bolshakov/bolshakov.h"
#include "systems/mishchenko/mishchenko.h"
#include "systems/vnm/vnm.h"
#include "systems/fitzhugh_nagumo/fitzhugh_nagumo.h"

extern std::map<std::string, Kernel> kernels;
extern std::map<std::string, int> kernelTPBs;
extern std::map<std::string, void(*)(Computation*)> kernelPrograms;
//extern std::map<std::string, void(*)(numb*, numb*, numb*)> kernelFDSs;
extern std::string selectedKernel;

#define addKernel(name)         kernels[#name] = readKernelText(#name); \
                                kernelTPBs[#name] = THREADS_PER_BLOCK_##name; \
                                kernelPrograms[#name] = kernelProgram_##name;// \
                                //kernelFDSs[#name] = finiteDifferenceScheme_##name;
#define selectKernel(name)      selectedKernel = #name;
#define KERNEL      kernels[selectedKernel]
#define KERNEL_TPB  kernelTPBs[selectedKernel]
#define KERNEL_PROG kernelPrograms[selectedKernel]
//#define KERNEL_FDS  kernelFDSs[selectedKernel]

int compute(Computation*);

// Fill trajectory and parameter buffers with initial values
void fillAttributeBuffers(Computation* data, int* attributeStepIndices, unsigned long long startVariation, unsigned long long endVariation, bool onlyTrajectory);

// Set and copy map buffers for continuous map computation
void setMapValues(Computation*);