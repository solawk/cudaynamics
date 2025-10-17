#pragma once
#include "variationSteps.h"

void extractMap(numb* src, numb* dst, int* indeces, int* steps, int axisXattr, int axisYattr, Kernel* kernel);

void setupLUT(numb* src, int particleCount, int** lut, int* groupSizes, int groupCount, numb min, numb max);