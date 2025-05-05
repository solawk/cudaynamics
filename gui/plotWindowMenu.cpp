#include "plotWindowMenu.h"

#define DONT_CLOSE_ON_CLICK_PUSH	ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
#define DONT_CLOSE_ON_CLICK_POP		ImGui::PopItemFlag();

void plotWindowMenu_File(PlotWindow* window);
void plotWindowMenu_PhasePlot(PlotWindow* window);
void plotWindowMenu_HeatmapPlot(PlotWindow* window);

extern bool enabledParticles;
extern bool autofitHeatmap;

void plotWindowMenu(PlotWindow* window)
{
	if (ImGui::BeginMenuBar())
	{
		plotWindowMenu_File(window);
		if (window->type == Phase || window->type == Phase2D) plotWindowMenu_PhasePlot(window);
		if (window->type == Heatmap || window->type == Phase2D) plotWindowMenu_HeatmapPlot(window);

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
		std::string windowName = window->name + std::to_string(window->id);

		plotWindowMenu_CommonPlot(window, windowName);

		bool tempShowHeatmapValues = window->hmp.showHeatmapValues; if (ImGui::Checkbox(("##" + windowName + "showHeatmapValues").c_str(), &tempShowHeatmapValues)) window->hmp.showHeatmapValues = !window->hmp.showHeatmapValues;
		ImGui::SameLine(); ImGui::Text("Show values");

		std::string diapasonsStrings[] = { "Values", "Steps" };
		bool tempShowActualDiapasons = window->hmp.showActualDiapasons;
		if (ImGui::BeginCombo(("##" + windowName + "diapasons").c_str(), (window->hmp.showActualDiapasons ? diapasonsStrings[0] : diapasonsStrings[1]).c_str()))
		{
			if (ImGui::Selectable(diapasonsStrings[0].c_str(), window->hmp.showActualDiapasons)) window->hmp.showActualDiapasons = true;
			if (ImGui::Selectable(diapasonsStrings[1].c_str(), !window->hmp.showActualDiapasons)) window->hmp.showActualDiapasons = false;
			ImGui::EndCombo();
		}

		bool tempHeatmapAutoCompute = window->hmp.isHeatmapAutoComputeOn; if (ImGui::Checkbox(("##" + windowName + "heatmapAutoCompute").c_str(), &tempHeatmapAutoCompute)) window->hmp.isHeatmapAutoComputeOn = !window->hmp.isHeatmapAutoComputeOn;
		ImGui::SameLine(); ImGui::Text("Auto-compute on Shift+RMB");

		ImGui::EndMenu();
	}
}