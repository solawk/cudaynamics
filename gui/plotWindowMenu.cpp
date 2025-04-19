#include "plotWindowMenu.h"

#define DONT_CLOSE_ON_CLICK_PUSH	ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
#define DONT_CLOSE_ON_CLICK_POP		ImGui::PopItemFlag();

void plotWindowMenu_File(PlotWindow* window);
void plotWindowMenu_PhasePlot(PlotWindow* window);

extern bool enabledParticles;

void plotWindowMenu(PlotWindow* window)
{
	if (ImGui::BeginMenuBar())
	{
		plotWindowMenu_File(window);
		if (window->type == Phase || window->type == Phase2D)
		{
			plotWindowMenu_PhasePlot(window);
		}

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

void plotWindowMenu_PhasePlot(PlotWindow* window)
{
	if (ImGui::BeginMenu("Plot"))
	{
		std::string windowName = window->name + std::to_string(window->id);

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