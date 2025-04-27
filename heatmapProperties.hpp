#pragma once

struct HeatmapProperties
{
	bool grayscaleHeatmap;
	int stride;

	float heatmapMax;
	float heatmapMin;
	bool areHeatmapLimitsDefined;
	bool isHeatmapDirty;
	void* myTexture;

	bool showHeatmapValues;
	bool showActualDiapasons;
	bool isHeatmapSelectionModeOn;
	bool isHeatmapAutoComputeOn;

	HeatmapProperties()
	{
		stride = 1;
		heatmapMax = 0.0f;
		heatmapMin = 0.0f;

		grayscaleHeatmap = false;
		isHeatmapSelectionModeOn = false;
		isHeatmapAutoComputeOn = false;
		areHeatmapLimitsDefined = false;

		isHeatmapDirty = false;
		myTexture = nullptr;

		showHeatmapValues = false;
		showActualDiapasons = true;
	}
};