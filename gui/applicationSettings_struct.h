#pragma once

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "customStylesEnum.h"

struct ApplicationSettings
{
	bool preciseNumbDrags;
	bool CPU_mode_interactive, CPU_mode_hires;
	bool calculateDeltaDecay;
	int threadsPerBlock;
	float dragChangeSpeed;

	ImVec4 cudaColor, openmpColor, hiresColor;

	ImGuiCustomStyle appStyle;

	ApplicationSettings()
	{
		preciseNumbDrags = false;

		CPU_mode_interactive = false;
		CPU_mode_hires = false;

		calculateDeltaDecay = true;
		threadsPerBlock = 32;

		dragChangeSpeed = 1.0f;

		cudaColor = ImVec4(0.40f, 0.56f, 0.18f, 1.00f);
		openmpColor = ImVec4(0.03f, 0.45f, 0.49f, 1.00f);
		hiresColor = ImVec4(0.50f, 0.10f, 0.30f, 1.00f);

		appStyle = ImGuiCustomStyle::Dark;
	}
};