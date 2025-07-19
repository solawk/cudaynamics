#pragma once

#define MAX_SETTINGS_PER_MAP 32

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

	int settingsCount;
	std::string settingName[MAX_SETTINGS_PER_MAP];
	bool isSettingNumb[MAX_SETTINGS_PER_MAP];
};