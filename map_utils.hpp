#pragma once
#include "variationSteps.h"

void extractMap(numb* src, numb* dst, int* steps, int axisXattr, int axisYattr, Kernel* kernel);

void setupLUT(numb* src, int particleCount, int** lut, int* groupSizes, int groupCount, float min, float max);