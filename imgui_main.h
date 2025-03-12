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

#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "main.h"

#include "implot/implot.h"
#include <tchar.h>
#include <objects.h>
#include <implot_internal.h>
#include "imgui_utils.h"
#include "resource.h"
#include "quaternion.h"

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int imgui_main(int, char**);

void listVariable(int i);
void listParameter(int i);