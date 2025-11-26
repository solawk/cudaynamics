#pragma once
#include "../imgui/imgui.h"
#include "../implot3d/implot3d.h"
#include "gui/customStylesEnum.h"

void SetupImGuiStyle(ImGuiCustomStyle cs, ImVec4 normal, ImVec4 hires, ImVec4 cpuMode, bool isHires, bool isCPU);