#pragma once

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "customStylesEnum.h"
#include "../json/json.h"
#include <fstream>
#include <string>
#include "../fontSettings_struct.h"
#include "../jsonRW.h"

struct ApplicationSettings
{
	static const bool preciseNumbDrags_default = false;
	static const bool CPU_mode_interactive_default = false;
	static const bool CPU_mode_hires_default = false;
	static const bool calculateDeltaDecay_default = true;
	static const int threadsPerBlock_default = 32;
	static const int varPerParallelization_default = 1000000;
	float dragChangeSpeed_default;
	ImVec4 cudaColor_default;
	ImVec4 openmpColor_default;
	ImVec4 hiresColor_default;
	static const ImGuiCustomStyle appStyle_default = ImGuiCustomStyle::Dark;

	bool preciseNumbDrags;
	bool CPU_mode_interactive, CPU_mode_hires;
	bool calculateDeltaDecay;
	int threadsPerBlock;
	int varPerParallelization;
	float dragChangeSpeed;
	ImVec4 cudaColor, openmpColor, hiresColor;
	ImGuiCustomStyle appStyle;
	FontSettings* globalFontSettings;

	ApplicationSettings()
	{
		dragChangeSpeed_default = 1.0f;
		cudaColor_default = ImVec4(0.40f, 0.56f, 0.18f, 1.00f);
		openmpColor_default = ImVec4(0.03f, 0.45f, 0.49f, 1.00f);
		hiresColor_default = ImVec4(0.50f, 0.10f, 0.30f, 1.00f);

		preciseNumbDrags = preciseNumbDrags_default;
		CPU_mode_interactive = CPU_mode_interactive_default;
		CPU_mode_hires = CPU_mode_hires_default;
		calculateDeltaDecay = calculateDeltaDecay_default;
		threadsPerBlock = threadsPerBlock_default;
		varPerParallelization = varPerParallelization_default;
		dragChangeSpeed = dragChangeSpeed_default;
		cudaColor = cudaColor_default;
		openmpColor = openmpColor_default;
		hiresColor = hiresColor_default;
		appStyle = appStyle_default;

		globalFontSettings = nullptr;
	}

	void Save()
	{
		json::jobject settings;
		settings["preciseNumbDrags"].set_boolean(preciseNumbDrags);
		settings["CPU_mode_interactive"].set_boolean(CPU_mode_interactive);
		settings["CPU_mode_hires"].set_boolean(CPU_mode_hires);
		settings["calculateDeltaDecay"].set_boolean(calculateDeltaDecay);
		settings["threadsPerBlock"] = threadsPerBlock;
		settings["varPerParallelization"] = varPerParallelization;
		settings["dragChangeSpeed"] = dragChangeSpeed;
		settings["cudaColor"] = std::vector<float>{ cudaColor.x, cudaColor.y, cudaColor.z, cudaColor.w };
		settings["openmpColor"] = std::vector<float>{ openmpColor.x, openmpColor.y, openmpColor.z, openmpColor.w };
		settings["hiresColor"] = std::vector<float>{ hiresColor.x, hiresColor.y, hiresColor.z, hiresColor.w };
		settings["appStyle"] = (int)appStyle;
		if (globalFontSettings)
		{
			settings["globalFontSize"] = globalFontSettings->size;
			settings["globalFontFamily"] = globalFontSettings->family;
			settings["globalFontIsBold"].set_boolean(globalFontSettings->isBold);
			settings["globalFontIsItalic"].set_boolean(globalFontSettings->isItalic);
		}

		JSONWrite(settings, "applicationSettings.json", false);
	}

	void Load()
	{
		json::jobject settings;
		if (!JSONRead("applicationSettings.json", &settings, false)) return;

		preciseNumbDrags = settings.has_key("preciseNumbDrags") ? settings["preciseNumbDrags"].is_true() : preciseNumbDrags_default;
		CPU_mode_interactive = settings.has_key("CPU_mode_interactive") ? settings["CPU_mode_interactive"].is_true() : CPU_mode_interactive_default;
		CPU_mode_hires = settings.has_key("CPU_mode_hires") ? settings["CPU_mode_hires"].is_true() : CPU_mode_hires_default;
		calculateDeltaDecay = settings.has_key("calculateDeltaDecay") ? settings["calculateDeltaDecay"].is_true() : calculateDeltaDecay_default;
		threadsPerBlock = settings.has_key("threadsPerBlock") ? (int)settings["threadsPerBlock"] : threadsPerBlock_default;
		varPerParallelization = settings.has_key("varPerParallelization") ? (int)settings["varPerParallelization"] : varPerParallelization_default;
		dragChangeSpeed = settings.has_key("dragChangeSpeed") ? (float)settings["dragChangeSpeed"] : dragChangeSpeed_default;
		appStyle = settings.has_key("appStyle") ? (ImGuiCustomStyle)(int)settings["appStyle"] : appStyle_default;

		cudaColor = settings.has_key("cudaColor") ?
			ImVec4(((std::vector<float>)settings["cudaColor"])[0], 
				((std::vector<float>)settings["cudaColor"])[1], 
				((std::vector<float>)settings["cudaColor"])[2], 
				((std::vector<float>)settings["cudaColor"])[3])
			: cudaColor_default;

		openmpColor = settings.has_key("openmpColor") ?
			ImVec4(((std::vector<float>)settings["openmpColor"])[0],
				((std::vector<float>)settings["openmpColor"])[1],
				((std::vector<float>)settings["openmpColor"])[2],
				((std::vector<float>)settings["openmpColor"])[3])
			: openmpColor_default;

		hiresColor = settings.has_key("hiresColor") ?
			ImVec4(((std::vector<float>)settings["hiresColor"])[0],
				((std::vector<float>)settings["hiresColor"])[1],
				((std::vector<float>)settings["hiresColor"])[2],
				((std::vector<float>)settings["hiresColor"])[3])
			: hiresColor_default;

		if (globalFontSettings)
		{
			if (settings.has_key("globalFontSize")) globalFontSettings->size = (int)settings["globalFontSize"];
			if (settings.has_key("globalFontFamily")) globalFontSettings->family = (int)settings["globalFontFamily"];
			if (settings.has_key("globalFontIsBold")) globalFontSettings->isBold = settings["globalFontIsBold"].is_true();
			if (settings.has_key("globalFontIsItalic")) globalFontSettings->isItalic = settings["globalFontIsItalic"].is_true();
		}
	}
};

extern ApplicationSettings applicationSettings;