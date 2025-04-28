#pragma once
#include "mapData_struct.h"

struct HeatmapProperties
{
	bool grayscaleHeatmap;
	int stride;

	float heatmapMax;
	float heatmapMin;
	bool areHeatmapLimitsDefined;
	void* myTexture;

	bool areValuesDirty;
	bool isHeatmapDirty;

	bool showHeatmapValues;
	bool showActualDiapasons;
	bool isHeatmapSelectionModeOn;
	bool isHeatmapAutoComputeOn;

	numb* valueBuffer;
	unsigned char* pixelBuffer;
	int lastBufferSize;

	int indexX;
	int indexY;
	MapDimensionType typeX;
	MapDimensionType typeY;

	HeatmapProperties()
	{
		stride = 1;
		heatmapMax = 0.0f;
		heatmapMin = 0.0f;

		grayscaleHeatmap = false;
		isHeatmapSelectionModeOn = false;
		isHeatmapAutoComputeOn = false;
		areHeatmapLimitsDefined = false;

		areValuesDirty = false;
		isHeatmapDirty = false;
		myTexture = nullptr;

		showHeatmapValues = false;
		showActualDiapasons = true;

		valueBuffer = nullptr;
		pixelBuffer = nullptr;
		lastBufferSize = -1;

		indexX = 0;
		indexY = 1;

		typeX = VARIABLE;
		typeY = VARIABLE;
	}
};