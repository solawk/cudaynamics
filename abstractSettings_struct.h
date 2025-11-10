#pragma once
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include <string>

struct AbstractAnalysisSettingsStruct
{
	bool userEnabled; // Map calculation is enabled by the user. If not, toCompute will be 100% false, if enabled, it will be considered
	bool toCompute;
	unsigned int valueCount; // Values per index

protected:
	void DisplayIntSetting(std::string name, int& setting)
	{
		ImGui::Text(name.c_str());
		ImGui::SameLine();
		ImGui::PushItemWidth(150.0f);

		int value = setting;
		ImGui::InputInt(("##Setting" + name).c_str(), &value);
		setting = value;

		ImGui::PopItemWidth();
	}

	void DisplayNumbSetting(std::string name, numb& setting)
	{
		ImGui::Text(name.c_str());
		ImGui::SameLine();
		ImGui::PushItemWidth(150.0f);

		double value = (double)setting;
		ImGui::InputDouble(("##Setting" + name).c_str(), &value, 0.0, 0.0, "%f");
		setting = (numb)value;

		ImGui::PopItemWidth();
	}

	void DisplayVarSetting(std::string name, int& setting)
	{
		// TODO: Redo into variable selection
		ImGui::Text(name.c_str());
		ImGui::SameLine();
		ImGui::PushItemWidth(150.0f);

		int value = setting;
		ImGui::InputInt(("##Setting" + name).c_str(), &value);
		setting = value;

		ImGui::PopItemWidth();
	}
};