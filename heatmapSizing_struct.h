#pragma once
#include "kernel_struct.h"
#include "objects.h"
#include "mapData_struct.h"

struct HeatmapSizing
{
	Kernel* kernel = nullptr;
	HeatmapProperties* hmp = nullptr;

	numb minX, stepX, maxX;
	numb minY, stepY, maxY;

	int xSize, ySize;

	void loadPointers(Kernel* krn, HeatmapProperties* hmpd)
	{
		kernel = krn;
		hmp = hmpd;
	}

	void initValues()
	{
		xSize = hmp->typeX == VARIABLE ? kernel->variables[hmp->indexX].TrueStepCount() : kernel->parameters[hmp->indexX].TrueStepCount();
		ySize = hmp->typeY == VARIABLE ? kernel->variables[hmp->indexY].TrueStepCount() : kernel->parameters[hmp->indexY].TrueStepCount();

		switch (hmp->typeX)
		{
		case PARAMETER:
			minX = kernel->parameters[hmp->indexX].min;
			stepX = kernel->parameters[hmp->indexX].step;
			maxX = kernel->parameters[hmp->indexX].max;
			break;
		case VARIABLE:
			minX = kernel->variables[hmp->indexX].min;
			stepX = kernel->variables[hmp->indexX].step;
			maxX = kernel->variables[hmp->indexX].max;
			break;
		case STEP: // TODO
			minX = 0;
			stepX = 0;
			maxX = 0;
			break;
		}

		switch (hmp->typeY)
		{
		case PARAMETER:
			minY = kernel->parameters[hmp->indexY].min;
			stepY = kernel->parameters[hmp->indexY].step;
			maxY = kernel->parameters[hmp->indexY].max;
			break;
		case VARIABLE:
			minY = kernel->variables[hmp->indexY].min;
			stepY = kernel->variables[hmp->indexY].step;
			maxY = kernel->variables[hmp->indexY].max;
			break;
		case STEP: // TODO
			minY = 0;
			stepY = 0;
			maxY = 0;
			break;
		}
	}

	ImVec4 plotRect; // Plot visible ranges
	int cutMinX, cutMaxX, cutMinY, cutMaxY; // Plotted steps
	int cutWidth, cutHeight; // Amount of plotted steps
	int stepCountX, stepCountY; // Total step counts of axis
	numb valueMinX, valueMaxX;
	numb valueMinY, valueMaxY;

	numb mapX1, mapX2;
	numb mapY1, mapY2;
	numb mapX1Cut, mapX2Cut;
	numb mapY1Cut, mapY2Cut;

	void initCutoff(float plotx0, float ploty0, float plotx1, float ploty1, bool actualDiapasons)
	{
		plotRect = ImVec4(plotx0, ploty0, plotx1, ploty1);

		if (!actualDiapasons)
		{
			// Step diapasons
			cutMinX = (int)floor(plotRect.x) - 1;    if (cutMinX < 0) cutMinX = 0;
			cutMinY = (int)floor(plotRect.y) - 1;    if (cutMinY < 0) cutMinY = 0;
			cutMaxX = (int)ceil(plotRect.z);         if (cutMaxX > xSize - 1) cutMaxX = xSize - 1;
			cutMaxY = (int)ceil(plotRect.w);         if (cutMaxY > ySize - 1) cutMaxY = ySize - 1;
		}
		else
		{
			// Value diapasons
			stepCountX = xSize;
			stepCountY = ySize;

			cutMinX = stepFromValue(minX, stepX, plotRect.x);    if (cutMinX < 0) cutMinX = 0;
			cutMinY = stepFromValue(minY, stepY, plotRect.y);    if (cutMinY < 0) cutMinY = 0;
			cutMaxX = stepFromValue(minX, stepX, plotRect.z);    if (cutMaxX > stepCountX - 1) cutMaxX = stepCountX - 1;
			cutMaxY = stepFromValue(minY, stepY, plotRect.w);    if (cutMaxY > stepCountY - 1) cutMaxY = stepCountY - 1;

			valueMinX = calculateValue(minX, stepX, cutMinX);
			valueMinY = calculateValue(minY, stepY, cutMinY);
			valueMaxX = calculateValue(minX, stepX, cutMaxX + 1);
			valueMaxY = calculateValue(minY, stepY, cutMaxY + 1);
		}

		cutWidth = cutMaxX - cutMinX + 1;
		cutHeight = cutMaxY - cutMinY + 1;

		mapX1 = actualDiapasons ? minX : 0;
		mapX2 = actualDiapasons ? maxX + stepX : xSize;
		mapY1 = actualDiapasons ? minY : 0;
		mapY2 = actualDiapasons ? maxY + stepY : ySize;

		mapX1Cut = actualDiapasons ? valueMinX : cutMinX;
		mapX2Cut = actualDiapasons ? valueMaxX : cutMaxX + 1;
		mapY1Cut = actualDiapasons ? valueMaxY : cutMaxY + 1;
		mapY2Cut = actualDiapasons ? valueMinY : cutMinY;
	}
};