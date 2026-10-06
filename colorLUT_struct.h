#pragma once

struct colorLUT
{
public:
	int lutGroups;
	int** lut;
	int* lutSizes;
	// For categorical maps (for example BoA), stores the original category
	// represented by every LUT group. Continuous heatmaps leave this null.
	int* groupLabels;

	colorLUT()
	{
		lutGroups = 1;
		lut = nullptr;
		lutSizes = nullptr;
		groupLabels = nullptr;
	}

	void Clear()
	{
		if (lut != nullptr)
		{
			for (int i = 0; i < lutGroups; i++)
			{
				if (lut[i] != nullptr)
				{
					delete[] lut[i];
				}
			}

			delete[] lut;
			lut = nullptr;
		}

		if (lutSizes != nullptr)
		{
			delete[] lutSizes;
			lutSizes = nullptr;
		}

		if (groupLabels != nullptr)
		{
			delete[] groupLabels;
			groupLabels = nullptr;
		}

		lutGroups = 0;
	}
};
