#pragma once
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include <string>
#include <vector>
#include "attribute_struct.h"
#include "anfuncs.h"
#include "anfunc_names.h"

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

	void DisplayVarSetting(std::string name, int& setting, std::vector<Attribute>& variables, bool allowMinusOne = false)
	{
		ImGui::Text(name.c_str());
		ImGui::SameLine();
		ImGui::PushItemWidth(150.0f);

		if (ImGui::BeginCombo(("##Setting_" + name).c_str(), setting == -1 ? "-" : (variables[setting].name).c_str()))
		{
			int varCount = (int)variables.size();
			for (int v = allowMinusOne ? -1 : 0; v < varCount; v++)
			{
				bool isSelected = setting == v;
				ImGuiSelectableFlags selectableFlags = 0;
				if (ImGui::Selectable(v == -1 ? "-" : variables[v].name.c_str(), isSelected, selectableFlags))
				{
					setting = v;
				}
			}

			ImGui::EndCombo();
		}

		ImGui::PopItemWidth();
	}

	bool isMapSetupOfCorrectLength(std::vector<std::string> setup, int targetLength)
	{
		return setup.size() == targetLength;
	}

	int s2i(std::string s)
	{
		return atoi(s.c_str());
	}

	numb s2n(std::string s)
	{
		return (numb)atof(s.c_str());
	}
};