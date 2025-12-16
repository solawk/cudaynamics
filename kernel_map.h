#pragma once
#include <map>
#include <string>
#include "kernel_struct.h"
#include "computation_struct.h"

extern std::map<std::string, Kernel> kernels;
extern std::map<std::string, int> kernelTPBs;
extern std::map<std::string, void(*)(Computation*, uint64_t)> kernelPrograms;
//extern std::map<std::string, void(*)(Computation*, uint64_t)> kernel_cpu;
extern std::map<std::string, void(*)(Computation*, uint64_t)> kernelWrappers;
extern std::map<std::string, void(*)(numb*, numb*, numb*)> kernelFDS;
extern std::string selectedKernel;