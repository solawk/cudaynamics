#include "kernel_map.h"

std::map<std::string, Kernel> kernels;
std::map<std::string, int> kernelTPBs;
std::map<std::string, void(*)(Computation*)> kernelPrograms;
std::string selectedKernel;