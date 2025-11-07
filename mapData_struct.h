#pragma once

#define MAX_SETTINGS_PER_MAP 32

enum MapDimensionType { MDT_Variable, MDT_Parameter, MDT_Step };

struct MapData
{
	bool userEnabled; // Map calculation is enabled by the user. If not, toCompute will be 100% false, if enabled, it will be considered
	bool toCompute;
	unsigned int valueCount; // Values per map. Previously each variation could only have one value

	unsigned long int xSize;
	unsigned long int ySize;
	unsigned long int offset;

	std::string name;

	int settingsOffset;
	int settingsCount;
	std::string settingName[MAX_SETTINGS_PER_MAP];
	bool isSettingNumb[MAX_SETTINGS_PER_MAP];
};