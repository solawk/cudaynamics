#include "map_utils.hpp"

void extractMap(numb* src, numb* dst, int* indeces, int* steps, int axisXattr, int axisYattr, Kernel* kernel)
{
	bool isXparam = axisXattr >= kernel->VAR_COUNT;
	bool isYparam = axisYattr >= kernel->VAR_COUNT;

	int axisX = isXparam ? axisXattr - kernel->VAR_COUNT : axisXattr;
	int axisY = isYparam ? axisYattr - kernel->VAR_COUNT : axisYattr;

	int xCount = isXparam ? kernel->parameters[axisX].TrueStepCount() : kernel->variables[axisX].TrueStepCount();
	int yCount = isYparam ? kernel->parameters[axisY].TrueStepCount() : kernel->variables[axisY].TrueStepCount();

	int* localSteps = new int[kernel->VAR_COUNT + kernel->PARAM_COUNT];
	memcpy(localSteps, steps, sizeof(int) * (kernel->VAR_COUNT + kernel->PARAM_COUNT));
	uint64_t variation;

	for (int y = 0; y < yCount; y++)
		for (int x = 0; x < xCount; x++)
		{
			localSteps[axisXattr] = x;
			localSteps[axisYattr] = y;

			steps2Variation(&variation, localSteps, kernel);
			dst[y * xCount + x] = src[variation];
			indeces[y * xCount + x] = variation;
		}

	delete[] localSteps;
}

void setupLUT(numb* src, int particleCount, int** lut, int* groupSizes, int groupCount, numb min, numb max)
{
	numb* thresholds = new numb[groupCount];

	numb valueDiapasonPerGroup = (max - min) / groupCount;
	for (int g = 0; g < groupCount; g++)
	{
		groupSizes[g] = 0;
		thresholds[g] = max - valueDiapasonPerGroup * (groupCount - g - 1);
	}

	for (int i = 0; i < particleCount; i++)
	{
		for (int g = 0; g < groupCount; g++)
		{
			if (src[i] <= thresholds[g] || g == groupCount - 1)
			{
				lut[g][groupSizes[g]] = i;
				groupSizes[g] = groupSizes[g] + 1;
				break;
			}
		}
	}

	delete[] thresholds;
}