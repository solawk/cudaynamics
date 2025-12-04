#pragma once

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "implot.h"

#include "mapData_struct.h"
#include "colorLUT_struct.h"
#include "numb.h"

struct HeatmapValues
{
	numb* valueBuffer;
	numb heatmapMax;
	numb heatmapMin; 
	numb lastHovered; // Display value under mouse cursor
	bool hoveredAway; // Is ^ out of bounds
	numb lastSelected; // Display value of selected variation (by ranging or shift+click)
	int mapValueIndex;
	bool areHeatmapLimitsDefined;

	HeatmapValues()
	{
		valueBuffer = nullptr;
		areHeatmapLimitsDefined = false; 
		heatmapMax = 0.0f;
		heatmapMin = 0.0f;
		lastHovered = 0.0f;
		hoveredAway = false;
		lastSelected = 0.0f;
		mapValueIndex = 0;
	}
};

enum ValueDisplayMode { VDM_Never, VDM_OnlyOnShiftClick, VDM_Split, VDM_Always };

struct HeatmapProperties
{
	bool areValuesDirty; // Values have changed and heatmap should be updated
	bool isHeatmapDirty; // Values have not changed but its look has and should be updated

	bool showHeatmapValues;
	bool showActualDiapasons;
	bool isHeatmapSelectionModeOn;
	bool isHeatmapAutoComputeOn;

	// Default heatmaps (even hi-res ones) are not multichannel
	// When a heatmap is multichannel, it uses the "channel[]" values to make up an RGB image
	HeatmapValues values;
	bool isMultichannel;
	HeatmapValues channel[3];

	unsigned char* pixelBuffer;	// pixel RGBA values of the heatmap
	void* texture;
	int* indexBuffer;
	int lastBufferSize;

	colorLUT paintLUT;

	int indexX;
	int indexY;
	MapDimensionType typeX;
	MapDimensionType typeY;

	bool initClickedLocation;
	ImVec2 lastClickedLocation;
	bool showDragLines;
	bool showLegend;
	ImPlotColormap colormap;
	ValueDisplayMode valueDisplay;

	bool ignoreLimitsRecalculationOnSelection;
	bool ignoreNextLimitsRecalculation;

	HeatmapProperties()
	{
		isHeatmapSelectionModeOn = false;
		isHeatmapAutoComputeOn = false;

		areValuesDirty = true;
		isHeatmapDirty = true;
		texture = nullptr;

		showHeatmapValues = false;
		showActualDiapasons = true;

		pixelBuffer = nullptr;
		indexBuffer = nullptr;
		lastBufferSize = -1;

		indexX = 0;
		indexY = 1;

		typeX = MDT_Variable;
		typeY = MDT_Variable;

		isMultichannel = false;

		initClickedLocation = false;
		lastClickedLocation = ImVec2(0.0f, 0.0f);
		showDragLines = true;
		showLegend = true;
		colormap = ImPlotColormap_Jet;
		valueDisplay = VDM_Always;

		ignoreLimitsRecalculationOnSelection = false;
		ignoreNextLimitsRecalculation = false;
	}
};