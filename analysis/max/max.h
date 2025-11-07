#pragma once
#include "../analysis.h"
#include "../computation_struct.h"

__device__ void MAX(Computation* data, MAX_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset);