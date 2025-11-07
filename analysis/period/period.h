#pragma once
#include <vector>
#include <cmath>

#include "../analysis.h"
#include "../computation_struct.h"
#include "period_settings.h"

using namespace std;

__device__ void Period(Computation* data, DBSCAN_Settings settings, int variation, void(*finiteDifferenceScheme)(numb*, numb*, numb*), int offset_period, int offset_meanPeak, int offset_meanInterval);