#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "lle_settings.h"

__device__ void LLE(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset);