#pragma once
#include <vector>
#include <cmath>

#include "../analysis.h"
#include "../computation_struct.h"
#include "period_settings.h"

__device__ void Period(Computation* data, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*));