#include "map_utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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

void setupCategoricalLUT(numb* src, int particleCount, colorLUT* lut)
{
	lut->Clear();
	if (src == nullptr || particleCount <= 0) return;

	std::vector<int> labels;
	labels.reserve(particleCount);
	for (int i = 0; i < particleCount; ++i)
		labels.push_back(std::isfinite((double)src[i]) ? (int)src[i] : -2);

	std::sort(labels.begin(), labels.end());
	labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

	lut->lutGroups = (int)labels.size();
	lut->lut = new int* [lut->lutGroups];
	lut->lutSizes = new int[lut->lutGroups]{};
	lut->groupLabels = new int[lut->lutGroups];

	for (int g = 0; g < lut->lutGroups; ++g)
	{
		lut->groupLabels[g] = labels[g];
		lut->lut[g] = nullptr;
	}

	// Count first so categorical LUT memory is O(number of variations), not
	// O(number of basins * number of variations).
	for (int i = 0; i < particleCount; ++i)
	{
		const int label = std::isfinite((double)src[i]) ? (int)src[i] : -2;
		const int group = (int)(std::lower_bound(labels.begin(), labels.end(), label) - labels.begin());
		++lut->lutSizes[group];
	}

	for (int g = 0; g < lut->lutGroups; ++g)
		lut->lut[g] = new int[lut->lutSizes[g]];

	std::vector<int> positions(lut->lutGroups, 0);
	for (int i = 0; i < particleCount; ++i)
	{
		const int label = std::isfinite((double)src[i]) ? (int)src[i] : -2;
		const int group = (int)(std::lower_bound(labels.begin(), labels.end(), label) - labels.begin());
		lut->lut[group][positions[group]++] = i;
	}
}
