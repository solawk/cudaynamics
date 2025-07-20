#pragma once
#include "mapData_struct.h"
#include "colorLUT_struct.h"

struct HeatmapProperties
{
	bool grayscaleHeatmap;
	int stride; // deprecated

	numb heatmapMax;
	numb heatmapMin;
	bool areHeatmapLimitsDefined;
	void* texture;

	bool areValuesDirty; // Values have changed and heatmap should be updated
	bool isHeatmapDirty; // Values have not changed but its look has and should be updated

	bool showHeatmapValues;
	bool showActualDiapasons;
	bool isHeatmapSelectionModeOn;
	bool isHeatmapAutoComputeOn;

	numb* valueBuffer;			// values of the heatmap
	unsigned char* pixelBuffer;	// pixel RGBA values of the heatmap
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
		heatmapMax = 0.0f;
		heatmapMin = 0.0f;

		grayscaleHeatmap = false;
		isHeatmapSelectionModeOn = false;
		isHeatmapAutoComputeOn = false;
		areHeatmapLimitsDefined = false;

		areValuesDirty = true;
		isHeatmapDirty = true;
		texture = nullptr;

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

	/*void obtainAxis(MapData& mapData)
	{
		indexX = mapData.indexX;
		indexY = mapData.indexY;

		typeX = mapData.typeX;
		typeY = mapData.typeY;
	}*/
};