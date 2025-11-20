#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "max_settings.h"

__host__ __device__ void MAX(Computation* data, uint64_t variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*));