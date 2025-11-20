#pragma once
#include <vector>
#include <cmath>

#include "../analysis.h"
#include "../computation_struct.h"
#include "period_settings.h"

__host__ __device__ void Period(Computation* data, uint64_t variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*));