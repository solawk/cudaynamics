#pragma once
#include "../analysis.h"
#include "../computation_struct.h"
#include "max_settings.h"

__device__ void MAX(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*));