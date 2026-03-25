#pragma once
#include <random>
#include "curand_kernel.h"
#include "numb.h"

struct PerThread
{
	std::mt19937_64* randomCPUgen;
	std::normal_distribution<numb>* randomCPUdistrib;
	curandState* randomGPUstate;
};
