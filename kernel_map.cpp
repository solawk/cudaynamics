#include "kernel_map.h"

std::map<std::string, Kernel> kernels;
std::map<std::string, int> kernelTPBs;
std::map<std::string, void(*)(Computation*, uint64_t)> kernelPrograms;
//std::map<std::string, void(*)(Computation*, uint64_t)> kernel_cpu;
std::map<std::string, void(*)(Computation*, uint64_t)> kernelWrappers;
std::map<std::string, void(*)(numb*, numb*, numb*)> kernelFDS;
std::string selectedKernel;

Kernel kernelNew, kernelHiresNew; // Front-end for the kernels in the GUI
Kernel kernelHiresComputed; // Hi-res computation kernel buffer which has been sent to computation