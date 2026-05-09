#include "fullscreen_funcs.h"

// set position and size for fullcreen
void FullscreenActLogic(PlotWindow* plotWindow, ImVec2* fullscreenSize)
{
    if (plotWindow->isFullscreen && plotWindow->isFullscreenStarted)
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(*fullscreenSize);
        plotWindow->isFullscreenStarted = false;
    }

    if (plotWindow->isFullscreenEnd)
    {
        ImGui::SetNextWindowPos(plotWindow->originalPos);
        ImGui::SetNextWindowSize(plotWindow->originalSize);
        plotWindow->isFullscreenEnd = false;

    }
}


void FullscreenButtonPressLogic(PlotWindow* plotWindow, ImGuiWindow* window)
{
    if (!window->IsFullscreenButtonPressed) return;

    if (window->IsFullscreen)
    {
        plotWindow->originalPos = ImGui::GetWindowPos();
        plotWindow->originalSize = ImGui::GetWindowSize();
        plotWindow->isFullscreenStarted = true;
    }
    else
        plotWindow->isFullscreenEnd = true;

    plotWindow->isFullscreen = window->IsFullscreen;
    window->IsFullscreenButtonPressed = false;

}


void FullscreenMouseMovementLogic(PlotWindow* plotWindow, ImGuiWindow* window) 
{
    if (window->IsFullscreenButtonPressed) return;
    if (window->Size == ImVec2((float)GetSystemMetrics(SM_CXSCREEN), (float)GetSystemMetrics(SM_CYSCREEN)) && window->Pos == ImVec2(0, 0)) {
        window->IsFullscreen = true;
        plotWindow->isFullscreen = true;
    }
    else if (window->IsFullscreen && (window->Size != ImVec2((float)GetSystemMetrics(SM_CXSCREEN), (float)GetSystemMetrics(SM_CYSCREEN)) || window->Pos != ImVec2(0, 0))) {
        window->IsFullscreen = false;
        plotWindow->isFullscreen = false;
    }
}