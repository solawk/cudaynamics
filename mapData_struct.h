#pragma once

enum MapDimensionType { VARIABLE, PARAMETER, STEP };

struct MapData
{
	unsigned long int xSize;
	unsigned long int ySize;
	unsigned long int offset;

	std::string name;

	int indexX;
	MapDimensionType typeX;

	int indexY;
	MapDimensionType typeY;
};