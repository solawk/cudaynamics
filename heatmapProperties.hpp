#pragma once
#include "mapData_struct.h"
#include "colorLUT_struct.h"

struct HeatmapProperties
{
	bool grayscaleHeatmap;
	int stride; // deprecated

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
	int* indexBuffer;
	int lastBufferSize;

	colorLUT dynamicLUT;
	colorLUT staticLUT;

	int indexX;
	int indexY;
	MapDimensionType typeX;
	MapDimensionType typeY;

	bool initClickedLocation;
	ImVec2 lastClickedLocation;
	bool showDragLines;
	bool showLegend;

	bool ignoreLimitsRecalculationOnSelection;
	bool ignoreNextLimitsRecalculation;

	HeatmapProperties()
	{
		stride = 1;
		heatmapMax = 0.0f;
		heatmapMin = 0.0f;

		grayscaleHeatmap = false;
		isHeatmapSelectionModeOn = false;
		isHeatmapAutoComputeOn = false;
		areHeatmapLimitsDefined = false;

		areValuesDirty = true;
		isHeatmapDirty = true;
		myTexture = nullptr;

		showHeatmapValues = false;
		showActualDiapasons = true;

		valueBuffer = nullptr;
		pixelBuffer = nullptr;
		indexBuffer = nullptr;
		lastBufferSize = -1;

		indexX = 0;
		indexY = 1;

		typeX = VARIABLE;
		typeY = VARIABLE;

		initClickedLocation = false;
		lastClickedLocation = ImVec2(0.0f, 0.0f);
		showDragLines = true;
		showLegend = true;

		ignoreLimitsRecalculationOnSelection = false;
		ignoreNextLimitsRecalculation = false;
	}
};