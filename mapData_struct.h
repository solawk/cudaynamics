#pragma once

enum MapDimensionType { MDT_Variable, MDT_Parameter, MDT_Step };

struct MapData
{
	bool userEnabled; // Map calculation is enabled by the user. If not, toCompute will be 100% false, if enabled, it will be considered
	bool toCompute;
	unsigned int valueCount; // Values per map. Previously each variation could only have one value

	unsigned long int offset;

	std::string name;
};