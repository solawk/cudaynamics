#pragma once
#include "../imgui_main.hpp"
#include <windows.h>
#include <shellapi.h>

void FullscreenActLogic(PlotWindow* plotWindow, ImVec2* fullscreenSize);

void FullscreenButtonPressLogic(PlotWindow* plotWindow, ImGuiWindow* window);

void FullscreenMouseMovementLogic(PlotWindow* plotWindow, ImGuiWindow* window);