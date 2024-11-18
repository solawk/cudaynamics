#pragma once
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "D:\Windows Kits\10\Include\10.0.19041.0\um\d3d11.h"
#pragma comment( lib, "d3d11.lib" )

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int imgui_main(int, char**);