#include "fullscreen_funcs.h"

// set position and size for fullcreen
void FullscreenActLogic(PlotWindow* plotWindow, ImVec2* fullscreenSize)
{
    if (plotWindow->isFullscreen)
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(*fullscreenSize);
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
    }
    else
        plotWindow->isFullscreenEnd = true;

    plotWindow->isFullscreen = window->IsFullscreen;
    window->IsFullscreenButtonPressed = false;
}