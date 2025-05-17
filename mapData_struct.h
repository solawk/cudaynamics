#pragma once

enum MapDimensionType { VARIABLE, PARAMETER, STEP };

struct MapData
{
	bool toCompute;

	unsigned long int xSize;
	unsigned long int ySize;
	unsigned long int offset;

	std::string name;

	int indexX;
	MapDimensionType typeX;

	int indexY;
	MapDimensionType typeY;

	int settingsOffset;

	// Map can be calculated even with >2 varying attributes, as long as we fixate the rest with a certain value
	// The fixed values of the varying attributes are stored in these arrays
	// PRETTY MUCH UNUSED AND SHOULD BE DEPRECATED, IF I DON'T CHANGE MY MIND
	//numb varFixations[MAX_ATTRIBUTES];
	//numb paramFixations[MAX_ATTRIBUTES];
};