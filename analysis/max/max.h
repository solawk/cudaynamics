#pragma once
#include "../analysis.h"
#include "../computation_struct.h"

__device__ void MAX(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*));