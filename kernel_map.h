#pragma once
#include <map>
#include <string>
#include "kernel_struct.h"
#include "computation_struct.h"

extern std::map<std::string, Kernel> kernels;
extern std::map<std::string, int> kernelTPBs;
extern std::map<std::string, void(*)(Computation*)> kernelPrograms;
extern std::string selectedKernel;