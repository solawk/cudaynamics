#pragma once
#include <vector>
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "implot.h"
#include "../numb.h"

enum OrbitPlotType { OPT_Peak_Bifurcation, OPT_Interval_Bifurcation, OPT_Selected_Var_Section, OPT_Bifurcation_3D, OPT_COUNT };

struct OrbitProperties
{
	int xIndex;
	bool showParLines;
	OrbitPlotType type;
	float pointSize;
	float markerWidth;
	ImVec4 markerColor;
	bool invertedAxes;

	bool areValuesDirty;
	numb* bifAmps;
	numb* bifParamIndices;
	numb* bifIntervals;
	std::vector<int> lastAttributeValueIndices;
	int bifDotAmount;
	int prevTotalVariation;
	bool isAutoComputeOn;

	bool continuationButtonPressed;
	bool redrawContinuation;
	bool drawingContinuation;
	numb* continuationAmpsForward;
	numb* continuationAmpsBack;
	numb* continuationIntervalsBack;
	numb* continuationIntervalsForward;
	numb* continuationParamIndicesBack;
	numb* continuationParamIndicesForward;
	int bifDotAmountForward;
	int bifDotAmountBack;
	std::vector<int> lastAttributevalueindicesContinuations;

	float pointSizeForward;
	float pointSizeBack;
	ImPlotMarker dotShapeForward;
	ImPlotMarker dotShapeBack;
	ImVec4 dotColorForward;
	ImVec4 dotColorBack;

	OrbitProperties()
	{
		xIndex = 0;
		showParLines = true;
		type = OPT_Peak_Bifurcation;
		pointSize = 0.5f;
		markerColor = ImVec4(1.0f, 0.0f, 0.0f, 0.5f);
		markerWidth = 1;
		invertedAxes = false;
		areValuesDirty = true;
		bifAmps = NULL;
		bifParamIndices = NULL;
		bifIntervals = NULL;
		isAutoComputeOn = false;

		continuationButtonPressed = false;
		redrawContinuation = false;
		drawingContinuation = false;
		continuationAmpsBack = NULL;
		continuationAmpsForward = NULL;
		continuationIntervalsBack = NULL;
		continuationIntervalsForward = NULL;
		continuationParamIndicesBack = NULL;
		continuationParamIndicesForward = NULL;

		pointSizeForward = 0.5f;
		pointSizeBack = 0.5f;
		dotShapeForward = ImPlotMarker_Circle;
		dotShapeBack = ImPlotMarker_Circle;
		dotColorForward = ImVec4(0.6f, 1.0f, 0.6f, 1.0f);
		dotColorBack = ImVec4(0.6f, 0.6f, 1.0f, 1.0f);

		bifDotAmount = 0;
		bifDotAmountBack = 0;
		bifDotAmountForward = 0;
		prevTotalVariation = 0;
	}
};