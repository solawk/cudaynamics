#pragma once
#include "variationSteps.h"
#include "colorLUT_struct.h"

void extractMap(numb* src, numb* dst, int* indeces, int* steps, int axisXattr, int axisYattr, Kernel* kernel);

void setupLUT(numb* src, int particleCount, int** lut, int* groupSizes, int groupCount, numb min, numb max);

// Builds one group per exact integer category and remembers the category id.
// This lets categorical maps paint particles/trajectories with precisely the
// same color as their source pixels.
void setupCategoricalLUT(numb* src, int particleCount, colorLUT* lut);
