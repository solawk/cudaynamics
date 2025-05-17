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
};