#include "plotWindowMenu.h"

#define DONT_CLOSE_ON_CLICK_PUSH	ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
#define DONT_CLOSE_ON_CLICK_POP		ImGui::PopItemFlag();

void plotWindowMenu_File(PlotWindow* window);
void plotWindowMenu_PhasePlot(PlotWindow* window);
void plotWindowMenu_HeatmapPlot(PlotWindow* window);
void plotWindowMenu_HeatmapColors(PlotWindow* window);

extern bool enabledParticles;
extern bool autofitHeatmap;
extern PlotWindow* colorsLUTfrom;
extern int staticLUTsize;
extern int dynamicLUTsize;
extern PlotWindow* hiresHeatmapWindow;

void plotWindowMenu(PlotWindow* window)
{
	if (ImGui::BeginMenuBar())
	{
		plotWindowMenu_File(window);
		if (window->type == Phase || window->type == Phase2D) plotWindowMenu_PhasePlot(window);
		if (window->type == Heatmap || window->type == Phase2D) plotWindowMenu_HeatmapPlot(window);
		if (window->type == Heatmap) plotWindowMenu_HeatmapColors(window);

		ImGui::EndMenuBar();
	}
}

void plotWindowMenu_File(PlotWindow* window)
{
	if (ImGui::BeginMenu("File"))
	{
		if (ImGui::MenuItem("Export"))
		{
			printf("Export placeholder\n");
		}

		ImGui::EndMenu();
	}
}

void plotWindowMenu_CommonPlot(PlotWindow* window, std::string windowName)
{
	bool tempWhiteBg = window->whiteBg; if (ImGui::Checkbox(("##" + windowName + "whiteBG").c_str(), &tempWhiteBg)) window->whiteBg = !window->whiteBg;
	ImGui::SameLine(); ImGui::Text("White background");

	bool tempGrayscale = window->hmp.grayscaleHeatmap; if (ImGui::Checkbox(("##" + windowName + "grayscale").c_str(), &tempGrayscale)) window->hmp.grayscaleHeatmap = !window->hmp.grayscaleHeatmap;
	ImGui::SameLine(); ImGui::Text("Grayscale");
}

void plotWindowMenu_PhasePlot(PlotWindow* window)
{
	if (ImGui::BeginMenu("Plot"))
	{
		std::string windowName = window->name + std::to_string(window->id);

		plotWindowMenu_CommonPlot(window, windowName);

		if (enabledParticles)
		{
			ImGui::ColorEdit4(("##" + windowName + "_particleColor").c_str(), (float*)(&(window->markerColor)));		ImGui::SameLine(); ImGui::Text("Particle color");
			ImGui::DragFloat(("##" + windowName + "_particleSize").c_str(), &(window->markerSize), 0.1f);				ImGui::SameLine(); ImGui::Text("Particle size");
			ImGui::DragFloat(("##" + windowName + "_particleOutlineSize").c_str(), &(window->markerOutlineSize), 0.1f);	ImGui::SameLine(); ImGui::Text("Particle outline size");
			if (window->markerSize < 0.0f) window->markerSize = 0.0f;
		}
		else
		{
			ImGui::ColorEdit4(("##" + windowName + "_lineColor").c_str(), (float*)(&(window->plotColor)));		ImGui::SameLine(); ImGui::Text("Line color");
		}

		if (window->type == Phase)
		{
			bool tempIsI3d = window->isImplot3d; if (ImGui::Checkbox(("##" + windowName + "isI3D").c_str(), &tempIsI3d)) window->isImplot3d = !window->isImplot3d;
			ImGui::SameLine(); ImGui::Text("Use ImPlot3D");
		}

		ImGui::EndMenu();
	}
}

void plotWindowMenu_HeatmapPlot(PlotWindow* window)
{
	if (ImGui::BeginMenu("Plot"))
	{
		bool isHires = window == hiresHeatmapWindow;
		HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;

		std::string windowName = window->name + std::to_string(window->id);

		plotWindowMenu_CommonPlot(window, windowName);

		bool tempShowHeatmapValues = heatmap->showHeatmapValues; if (ImGui::Checkbox(("##" + windowName + "showHeatmapValues").c_str(), &tempShowHeatmapValues)) heatmap->showHeatmapValues = !heatmap->showHeatmapValues;
		ImGui::SameLine(); ImGui::Text("Show values");

		bool tempShowDragLines = heatmap->showDragLines; if (ImGui::Checkbox(("##" + windowName + "showDragLines").c_str(), &tempShowDragLines)) heatmap->showDragLines = !heatmap->showDragLines;
		ImGui::SameLine(); ImGui::Text("Show crosshair lines");

		bool tempShowLegend = heatmap->showDragLines; if (ImGui::Checkbox(("##" + windowName + "showLegend").c_str(), &tempShowLegend)) heatmap->showLegend = !heatmap->showLegend;
		ImGui::SameLine(); ImGui::Text("Show colormap");

		std::string diapasonsStrings[] = { "Values", "Steps" };
		bool tempShowActualDiapasons = heatmap->showActualDiapasons;
		if (ImGui::BeginCombo(("##" + windowName + "diapasons").c_str(), (heatmap->showActualDiapasons ? diapasonsStrings[0] : diapasonsStrings[1]).c_str()))
		{
			if (ImGui::Selectable(diapasonsStrings[0].c_str(), heatmap->showActualDiapasons)) heatmap->showActualDiapasons = true;
			if (ImGui::Selectable(diapasonsStrings[1].c_str(), !heatmap->showActualDiapasons)) heatmap->showActualDiapasons = false;
			ImGui::EndCombo();
		}

		bool tempHiresMode = window == hiresHeatmapWindow; if (ImGui::Checkbox(("##" + windowName + "isHiresMode").c_str(), &tempHiresMode))
		{
			if (window != hiresHeatmapWindow)
			{
				hiresHeatmapWindow = window;
			}
			else
			{
				hiresHeatmapWindow = nullptr;
			}
		}
		ImGui::SameLine(); ImGui::Text("Hi-Res mode");

		bool tempHeatmapAutoCompute = heatmap->isHeatmapAutoComputeOn; if (ImGui::Checkbox(("##" + windowName + "heatmapAutoCompute").c_str(), &tempHeatmapAutoCompute)) heatmap->isHeatmapAutoComputeOn = !heatmap->isHeatmapAutoComputeOn;
		ImGui::SameLine(); ImGui::Text("Auto-compute on Shift+RMB");

		bool tempIgnoreLimitsRecalc = heatmap->ignoreLimitsRecalculationOnSelection; if (ImGui::Checkbox(("##" + windowName + "heatmapignoreLimitsRecalc").c_str(), &tempIgnoreLimitsRecalc)) heatmap->ignoreLimitsRecalculationOnSelection = !heatmap->ignoreLimitsRecalculationOnSelection;
		ImGui::SameLine(); ImGui::Text("Auto-compute does not update limits");

		ImGui::EndMenu();
	}
}

void plotWindowMenu_HeatmapColors(PlotWindow* window)
{
	if (ImGui::BeginMenu("Painting"))
	{
		std::string windowName = window->name + std::to_string(window->id);

		if (colorsLUTfrom != window)
		{
			if (ImGui::Button("Use heatmap for painting"))
			{
				colorsLUTfrom = window;
			}
		}
		else
		{
			if (ImGui::Button("Stop using heatmap for painting"))
			{
				colorsLUTfrom = nullptr;
			}
		}

		int tempDLS = dynamicLUTsize;
		int tempSLS = staticLUTsize;

		ImGui::DragInt(("##" + windowName + "_dynamicLUT").c_str(), &(dynamicLUTsize));	ImGui::SameLine(); ImGui::Text("Colors when playing");
		ImGui::DragInt(("##" + windowName + "_staticLUT").c_str(), &(staticLUTsize));	ImGui::SameLine(); ImGui::Text("Colors when paused");

		if (dynamicLUTsize < 2) dynamicLUTsize = 2;
		if (staticLUTsize < 2) staticLUTsize = 2;

		if (tempDLS != dynamicLUTsize || tempSLS != staticLUTsize) window->hmp.isHeatmapDirty = true;

		ImGui::EndMenu();
	}
}