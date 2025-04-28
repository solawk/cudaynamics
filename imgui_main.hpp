#pragma once
#include "d3d11.h"
#pragma comment( lib, "d3d11.lib" )

#include <future>
#include <atomic>
#include <set>
#include <fstream>
#include <stdio.h>
#include <string>
#include <functional>
#include <vector>
#include <map>
#include <chrono>

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "main.h"

#include "implot/implot.h"
#include <tchar.h>
#include <objects.h>
#include <implot_internal.h>
#include "implot3d/implot3d_internal.h"
#include "imgui_utils.h"
#include "resource.h"
#include "quaternion.h"
#include "heatmapSizing_struct.h"
#include "map_utils.hpp"

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int imgui_main(int, char**);

void listVariable(int i);
void listParameter(int i);
void heatmapRangingSelection(PlotWindow* window, ImPlotPlot* plot, HeatmapSizing* sizing);

// ImGui macros to shorten the .cpp
#define IMGUI_WORK_BEGIN	MSG msg;                                                                                            \
                            while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))                                             \
                            {                                                                                                   \
                                ::TranslateMessage(&msg);                                                                       \
                                ::DispatchMessage(&msg);                                                                        \
                                if (msg.message == WM_QUIT)                                                                     \
                                    work = false;                                                                               \
                            }                                                                                                   \
                            if (!work)                                                                                          \
                            break;                                                                                              \
                                                                                                                                \
                            if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)     \
                            {                                                                                                   \
                                ::Sleep(10);                                                                                    \
                                continue;                                                                                       \
                            }                                                                                                   \
                            g_SwapChainOccluded = false;                                                                        \
                                                                                                                                \
                            if (g_ResizeWidth != 0 && g_ResizeHeight != 0)                                                      \
                            {                                                                                                   \
                                CleanupRenderTarget();                                                                          \
                                g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);          \
                                g_ResizeWidth = g_ResizeHeight = 0;                                                             \
                                CreateRenderTarget();                                                                           \
                            }                                                                                                   \
                                                                                                                                \
                            ImGui_ImplDX11_NewFrame();                                                                          \
                            ImGui_ImplWin32_NewFrame();                                                                         \
                            ImGui::NewFrame();

#define IMGUI_WORK_END      ImGui::Render();                                                                                                                                            \
                            ImVec4 clear_color = ImVec4(1.0f, 1.0f, 1.0f, 1.00f);                                                                                                       \
                            const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };     \
                            g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);                                                                               \
                            g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);                                                                 \
                            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());                                                                                                        \
                                                                                                                                                                                        \
                            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)                                                                                                      \
                            {                                                                                                                                                           \
                                ImGui::UpdatePlatformWindows();                                                                                                                         \
                                ImGui::RenderPlatformWindowsDefault();                                                                                                                  \
                            }                                                                                                                                                           \
                                                                                                                                                                                        \
                            HRESULT hr = g_pSwapChain->Present(1, 0);                                                                                                                   \
                            g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);