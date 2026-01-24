#pragma once
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"

// Translate-Rotation-Scale (for phase plots)
struct TRS
{
	ImVec4 offset;
	ImVec4 scale;
	ImVec4 quatRot;
	ImVec4 autorotate; // euler angles
	ImVec2 deltarotation; // euler angles

	TRS()
	{
		offset = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		scale = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

		quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
		autorotate = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		deltarotation = ImVec2(0.0f, 0.0f);
	}
};