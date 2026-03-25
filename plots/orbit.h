#pragma once
#include <vector>
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "implot.h"
#include "../numb.h"
#include "../kernel_map.h"
#include "../variationSteps.h"

enum OrbitPlotType { OPT_Peak_Bifurcation, OPT_Interval_Bifurcation, OPT_Selected_Var_Section, OPT_Bifurcation_3D, OPT_COUNT };

constexpr int MAX_PEAKS = 1024;

struct OrbitProperties
{
	int xIndex;
	int analyzedVariable;
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

	numb minX, maxX, minY, maxY;
	int peakCount;
	std::vector<numb> SliceForwardInt;
	std::vector<numb> SliceForwardPeak;
	std::vector<numb> SliceBackwardInt;
	std::vector<numb> SliceBackwardPeak;
	int forNum;
	int backNum;

	numb* peakAmplitudes;
	numb* peakIntervals;

	float pointSizeForward;
	float pointSizeBack;
	ImPlotMarker dotShapeForward;
	ImPlotMarker dotShapeBack;
	ImVec4 dotColorForward;
	ImVec4 dotColorBack;

	OrbitProperties()
	{
		xIndex = 0;
		analyzedVariable = 0;
		showParLines = true;
		type = OPT_Peak_Bifurcation;
		pointSize = 0.5f;
		markerColor = ImVec4(1.0f, 0.0f, 0.0f, 0.5f);
		markerWidth = 1;
		invertedAxes = false;
		areValuesDirty = true;
		bifAmps = bifParamIndices = bifIntervals = NULL;
		isAutoComputeOn = false;

		continuationButtonPressed = false;
		redrawContinuation = false;
		drawingContinuation = false;
		continuationAmpsBack = continuationAmpsForward = continuationIntervalsBack = continuationIntervalsForward = continuationParamIndicesBack = continuationParamIndicesForward = NULL;

		pointSizeForward = 0.5f;
		pointSizeBack = 0.5f;
		dotShapeForward = ImPlotMarker_Circle;
		dotShapeBack = ImPlotMarker_Circle;
		dotColorForward = ImVec4(0.6f, 1.0f, 0.6f, 1.0f);
		dotColorBack = ImVec4(0.6f, 0.6f, 1.0f, 1.0f);

		bifDotAmount = bifDotAmountBack = bifDotAmountForward = prevTotalVariation = 0;
		minX = maxX = minY = maxY = 0;
		peakCount = forNum = backNum = 0;

		peakAmplitudes = peakIntervals = nullptr;
	}

	void Continuation(int variationSize, Attribute* axis, std::vector<int>& attributeValueIndices)
	{
		if (redrawContinuation || continuationButtonPressed)
		{
			numb step = KERNEL.GetStepSize();
			int varCount = KERNEL.VAR_COUNT;
			int paramCount = KERNEL.PARAM_COUNT;

			lastAttributevalueindicesContinuations = attributeValueIndices;

			PerThread pt;
			std::mt19937_64 gen(123U);
			std::normal_distribution<numb> distrib(0.0, 1.0);
			pt.randomCPUgen = &gen;
			pt.randomCPUdistrib = &distrib;

			numb* startingVariables = new numb[varCount];
			numb* afterTrstartingVariables = new numb[varCount];
			numb* newVariables = new numb[varCount];
			numb* parameters = new numb[paramCount];
			continuationParamIndicesBack = new numb[MAX_PEAKS * axis->stepCount];
			continuationParamIndicesForward = new numb[MAX_PEAKS * axis->stepCount];
			continuationAmpsBack = new numb[MAX_PEAKS * axis->stepCount];
			continuationAmpsForward = new numb[MAX_PEAKS * axis->stepCount];
			continuationIntervalsBack = new numb[MAX_PEAKS * axis->stepCount];
			continuationIntervalsForward = new numb[MAX_PEAKS * axis->stepCount];
			std::vector<numb> trajectory;
			for (int i = 0; i < varCount; i++) {
				startingVariables[i] = attributeValueIndices[i] == 0 ? KERNEL.variables[i].min : KERNEL.variables[i].values[attributeValueIndices[i]];
			}
			for (int i = 0; i < paramCount; i++) 
			{
				if (KERNEL.parameters[i].rangingType != RT_Enum)
				{
					parameters[i] = attributeValueIndices[i + varCount] == 0 ? KERNEL.parameters[i].min : KERNEL.parameters[i].values[attributeValueIndices[i + varCount]];
				}
				else
				{
					parameters[i] = KERNEL.parameters[i].values[attributeValueIndices[i + varCount]];
				}
			}

			parameters[xIndex] = KERNEL.parameters[xIndex].min;
			for (int i = 0; i < KERNEL.transientSteps; i++) { kernelFDS[selectedKernel](startingVariables, newVariables, parameters, &pt); for (int j = 0; j < varCount; j++) startingVariables[j] = newVariables[j]; }
			for (int i = 0; i < varCount; i++) afterTrstartingVariables[i] = startingVariables[i];
			trajectory.push_back(startingVariables[analyzedVariable]);

			int BifDotAmount = 0;

			for (int j = 0; j < axis->stepCount; j++) {
				//for (int i = 0; i < varCount; i++) startingVariables[i] = afterTrstartingVariables[i];
				//trajectory.push_back(startingVariables[analyzedVariable]);
				parameters[xIndex] = KERNEL.parameters[xIndex].values[j];
				for (int trajstep = 0; trajstep < variationSize / varCount; trajstep++) {
					kernelFDS[selectedKernel](startingVariables, newVariables, parameters, &pt);
					trajectory.push_back(newVariables[analyzedVariable]);
					for (int i = 0; i < varCount; i++) startingVariables[i] = newVariables[i];
				}
				int peakCount = 0;
				bool firstpeakreached = false;
				numb temppeakindex;
				for (int trajstep = 1; trajstep < variationSize / varCount - 1 && peakCount < MAX_PEAKS; trajstep++) {
					numb prev = trajectory[trajstep - 1];
					numb curr = trajectory[trajstep];
					numb next = trajectory[trajstep + 1];
					if (curr > prev && curr > next)
					{
						if (firstpeakreached == false)
						{
							firstpeakreached = true;
							temppeakindex = (float)trajstep;
						}
						else
						{
							continuationAmpsForward[BifDotAmount] = curr;
							continuationIntervalsForward[BifDotAmount] = (trajstep - temppeakindex) * step;
							continuationParamIndicesForward[BifDotAmount] = axis->values[j];
							temppeakindex = (float)trajstep;
							peakCount++;
							BifDotAmount++;
						}
					}
				}
				trajectory.clear();
			}
			bifDotAmountForward = BifDotAmount;


			for (int i = 0; i < varCount; i++) {
				startingVariables[i] = attributeValueIndices[i] == 0 ? KERNEL.variables[i].min : KERNEL.variables[i].values[attributeValueIndices[i]];
			}

			parameters[xIndex] = KERNEL.parameters[xIndex].max;
			for (int i = 0; i < KERNEL.transientSteps; i++) { kernelFDS[selectedKernel](startingVariables, newVariables, parameters, &pt); for (int j = 0; j < varCount; j++) startingVariables[j] = newVariables[j]; }
			for (int i = 0; i < varCount; i++) afterTrstartingVariables[i] = startingVariables[i];
			trajectory.push_back(startingVariables[analyzedVariable]);

			BifDotAmount = 0;
			for (int j = axis->stepCount - 1; j >= 0; j--) {
				//for (int i = 0; i < varCount; i++) startingVariables[i] = afterTrstartingVariables[i];
				//trajectory.push_back(startingVariables[analyzedVariable]);
				parameters[xIndex] = KERNEL.parameters[xIndex].values[j];
				for (int trajstep = 0; trajstep < variationSize / varCount; trajstep++) {
					kernelFDS[selectedKernel](startingVariables, newVariables, parameters, &pt);
					trajectory.push_back(newVariables[analyzedVariable]);
					for (int i = 0; i < varCount; i++) startingVariables[i] = newVariables[i];
				}
				int peakCount = 0;
				bool firstpeakreached = false;
				numb temppeakindex;
				for (int trajstep = 1; trajstep < variationSize / varCount - 1 && peakCount < MAX_PEAKS; trajstep++) {
					numb prev = trajectory[trajstep - 1];
					numb curr = trajectory[trajstep];
					numb next = trajectory[trajstep + 1];

					if (curr > prev && curr > next)
					{
						if (firstpeakreached == false)
						{
							firstpeakreached = true;
							temppeakindex = (float)trajstep;
						}
						else
						{
							continuationAmpsBack[BifDotAmount] = curr;
							continuationIntervalsBack[BifDotAmount] = (trajstep - temppeakindex) * step;
							continuationParamIndicesBack[BifDotAmount] = axis->values[j];
							temppeakindex = (float)trajstep;
							peakCount++;
							BifDotAmount++;
						}
					}
				}
				trajectory.clear();
			}
			bifDotAmountBack = BifDotAmount;

			redrawContinuation = false;
			drawingContinuation = true;
			continuationButtonPressed = false;
		}
	}

	void Calculation(int variationSize, Attribute* axis, std::vector<int>& attributeValueIndices, bool OrbitRedraw, numb* trajectory)
	{
		uint64_t variation;

		numb step = KERNEL.GetStepSize();
		int varCount = KERNEL.VAR_COUNT;
		int paramCount = KERNEL.PARAM_COUNT;

		if (OrbitRedraw && peakAmplitudes != nullptr)
		{
			delete[] peakAmplitudes;
			delete[] peakIntervals;

			peakAmplitudes = nullptr;
			peakIntervals = nullptr;
		}

		if (peakAmplitudes == nullptr)
		{
			peakAmplitudes = new numb[MAX_PEAKS];
			peakIntervals = new numb[MAX_PEAKS];
		}

		if (type == OPT_Selected_Var_Section)
		{
			steps2Variation(&variation, &(attributeValueIndices.data()[0]), &KERNEL);

			peakCount = 0;
			bool firstpeakreached = false;
			numb temppeakindex;
			numb* computedVariation = trajectory + (variationSize * variation);
			for (int i = 1; i < variationSize / varCount - 1 && peakCount < MAX_PEAKS; i++)
			{
				numb prev = computedVariation[analyzedVariable + varCount * i - varCount];
				numb curr = computedVariation[analyzedVariable + varCount * i];
				numb next = computedVariation[analyzedVariable + varCount * i + varCount];
				if (curr > prev && curr > next)
				{
					if (firstpeakreached == false)
					{
						firstpeakreached = true;
						temppeakindex = (float)i;
					}
					else
					{

						peakAmplitudes[peakCount] = curr;
						peakIntervals[peakCount] = (i - temppeakindex) * step;
						peakCount++;
						temppeakindex = (float)i;
					}
				}
			}

			minX = peakIntervals[0];
			maxX = peakIntervals[0];
			minY = peakAmplitudes[0];
			maxY = peakAmplitudes[0];

			for (int i = 0; i < peakCount - 1; ++i)
			{
				if (peakIntervals[i] < minX) minX = peakIntervals[i];
				if (peakIntervals[i] > maxX) maxX = peakIntervals[i];
				if (peakAmplitudes[i + 1] < minY) minY = peakAmplitudes[i + 1];
				if (peakAmplitudes[i + 1] > maxY) maxY = peakAmplitudes[i + 1];
			}

			forNum = 0;
			backNum = 0;
			bool sectionFound = false;
			if (drawingContinuation && continuationAmpsForward != NULL || lastAttributevalueindicesContinuations != attributeValueIndices && continuationAmpsForward != NULL) {
				SliceBackwardInt.clear(); SliceBackwardPeak.clear(); SliceForwardInt.clear(); SliceForwardPeak.clear();
				lastAttributevalueindicesContinuations = attributeValueIndices;
				for (int i = 0; i < bifDotAmountForward - 1; i++) {
					if (continuationParamIndicesForward[i] == axis->values[attributeValueIndices[varCount + xIndex]]) {
						if (!sectionFound) { sectionFound = true; }
						SliceForwardInt.push_back(continuationIntervalsForward[i]); SliceForwardPeak.push_back(continuationAmpsForward[i]);
						forNum++;
					}
				}
				for (int i = 0; i < bifDotAmountBack - 1; i++) {
					if (continuationParamIndicesBack[i] == axis->values[attributeValueIndices[varCount + xIndex]]) {
						if (!sectionFound) { sectionFound = true; }
						SliceBackwardInt.push_back(continuationIntervalsBack[i]); SliceBackwardPeak.push_back(continuationAmpsBack[i]);
						backNum++;
					}
				}
			}
		}
		else
		{
			if (OrbitRedraw) areValuesDirty = true;

			if (lastAttributeValueIndices.size() != 0)
			{
				for (int i = 0; i < varCount + paramCount; i++) 
				{
					if (i != varCount + xIndex) 
					{
						if (attributeValueIndices[i] != lastAttributeValueIndices[i])
						{
							areValuesDirty = true;
						}
					}
				}
			}

			if (areValuesDirty)
			{
				if (bifAmps != nullptr) { delete[]bifAmps; delete[]bifIntervals; delete[]bifParamIndices; }
				bifAmps = new numb[MAX_PEAKS * axis->stepCount]{ 0.0 };
				bifIntervals = new numb[MAX_PEAKS * axis->stepCount]{ 0.0 };
				bifParamIndices = new numb[MAX_PEAKS * axis->stepCount]{ 0.0 };
				std::vector<int> tempattributeValueIndices = attributeValueIndices;
				lastAttributeValueIndices = attributeValueIndices;
				int BifDotAmount = 0;
				for (int j = 0; j < axis->stepCount; j++)
				{
					tempattributeValueIndices[xIndex + varCount] = j;
					steps2Variation(&variation, &(tempattributeValueIndices.data()[0]), &KERNEL);
					numb* computedVariation = trajectory + (variationSize * variation);
					int peakCount = 0;
					bool firstpeakreached = false;
					numb temppeakindex;
					for (int i = 1; i < variationSize / varCount - 1 && peakCount < MAX_PEAKS; i++)
					{
						numb prev = computedVariation[analyzedVariable + varCount * i - varCount];
						numb curr = computedVariation[analyzedVariable + varCount * i];
						numb next = computedVariation[analyzedVariable + varCount * i + varCount];

						if (curr > prev && curr > next)
						{
							if (firstpeakreached == false)
							{
								firstpeakreached = true;
								temppeakindex = (float)i;
							}
							else
							{
								bifAmps[BifDotAmount] = curr;
								bifIntervals[BifDotAmount] = (i - temppeakindex) * step;
								bifParamIndices[BifDotAmount] = axis->values[j];
								temppeakindex = (float)i;
								peakCount++;
								BifDotAmount++;

							}
						}
					}

				}
				bifDotAmount = BifDotAmount;
				areValuesDirty = false;
			}

			if (type == OPT_Peak_Bifurcation)
			{
				minX = bifParamIndices[0];
				maxX = bifParamIndices[0];
				minY = bifIntervals[0];
				maxY = bifIntervals[0];

				for (int i = 0; i < bifDotAmount - 1; ++i)
				{
					maxX = bifParamIndices[i];
					if (bifAmps[i + 1] < minY) minY = bifAmps[i + 1];
					if (bifAmps[i + 1] > maxY) maxY = bifAmps[i + 1];
				}
			}
			else if (type == OPT_Interval_Bifurcation)
			{
				minX = bifParamIndices[0];
				maxX = bifParamIndices[0];
				minY = bifIntervals[0];
				maxY = bifIntervals[0];

				for (int i = 0; i < bifDotAmount - 1; ++i)
				{
					maxX = bifParamIndices[i];
					if (bifIntervals[i + 1] < minY) minY = bifIntervals[i + 1];
					if (bifIntervals[i + 1] > maxY) maxY = bifIntervals[i + 1];
				}

			}
			else if (type == OPT_Bifurcation_3D)
			{

			}
		}
	}
};