#pragma once

struct colorLUT
{
public:
	int lutGroups;
	int** lut;
	int* lutSizes;

	colorLUT()
	{
		lutGroups = 1;
		lut = nullptr;
		lutSizes = nullptr;
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
	}
};