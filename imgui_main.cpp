#include "imgui_main.h"
#include "implot/implot.h"
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <objects.h>
#include <vector>
#include <implot_internal.h>
#include "imgui_utils.h"

// Data
static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

// Main code
int imgui_main(int, char**)
{
    // Create application window
    //ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"ImGui Example", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"Dear ImGui DirectX11 Example", WS_OVERLAPPEDWINDOW, 100, 100, 300, 50, nullptr, nullptr, wc.hInstance, nullptr);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    //::ShowWindow(hwnd, SW_HIDE);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    io.ConfigViewportsNoAutoMerge = true;
    io.ConfigViewportsNoTaskBarIcon = true;
    //io.ConfigViewportsNoDefaultParent = true;
    //io.ConfigDockingAlwaysTabBar = true;
    //io.ConfigDockingTransparentPayload = true;
    //io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;     // FIXME-DPI: Experimental. THIS CURRENTLY DOESN'T WORK AS EXPECTED. DON'T USE IN USER APP!
    //io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleViewports; // FIXME-DPI: Experimental.

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    io.Fonts->AddFontFromFileTTF("C:\\Users\\Alexander\\AppData\\Local\\Microsoft\\Windows\\Fonts\\UbuntuMono-R.ttf", 24.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    int uniqueIds = 0;

    void* computedData = nullptr;
    void* dataBuffer = nullptr;
    void* axisBuffer = new float[3 * 6] {}; // 3 axis, 6 points;
    int computedSteps = 0;
    bool autofitAfterComputing = false;
    bool showComputedPlot = false; // TEMP
    PostRanging rangingData;
    if (kernel::executeOnLaunch)
    {
        if (computedData) delete[] (float*)computedData;
        compute(&computedData, &rangingData);
        showComputedPlot = true;
        computedSteps = kernel::steps;
        if (dataBuffer) delete[] dataBuffer;
        dataBuffer = new float[(computedSteps + 1) * kernel::VAR_COUNT];
    }

    float DEG2RAD = 3.141592f / 180.0f;

    // Main loop
    bool work = true;

    std::vector<PlotWindow> plotWindows;
    char* plotNameBuffer = new char[64]();
    strcpy(plotNameBuffer, "Plot");

    int selectedPlotVars[3]; selectedPlotVars[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVars[i] = -1;

    int variation = 0;
    int stride = 1;

    while (work)
    {
        // Poll and handle messages (inputs, window resize, etc.)
        // See the WndProc() function below for our to dispatch events to the Win32 backend.
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                work = false;
        }
        if (!work)
            break;

        // Handle window being minimized or screen locked
        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
        {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        // Handle window resize (we don't resize directly in the WM_SIZE handler)
        if (g_ResizeWidth != 0 && g_ResizeHeight != 0)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // MAIN WINDOW
        {
            style.WindowMenuButtonPosition = ImGuiDir_Left;
            ImGui::Begin("CUDAynamics", &work);
            ImGui::Text(kernel::name);

            // Parameters & Variables

            char* NAME_PADDED;
            int maxNameLength = 0;

            for (int i = 0; i < kernel::PARAM_COUNT; i++) if (strlen(kernel::PARAM_NAMES[i]) > maxNameLength) maxNameLength = (int)strlen(kernel::PARAM_NAMES[i]);
            for (int i = 0; i < kernel::VAR_COUNT; i++) if (strlen(kernel::VAR_NAMES[i]) > maxNameLength) maxNameLength = (int)strlen(kernel::VAR_NAMES[i]);          

            ImGui::SeparatorText("Variables");

            for (int i = 0; i < kernel::VAR_COUNT; i++)
            {
                NAME_PADDED = new char[maxNameLength + 1];
                strcpy(NAME_PADDED, kernel::VAR_NAMES[i]);
                for (int j = (int)strlen(kernel::VAR_NAMES[i]); j < maxNameLength; j++)
                    NAME_PADDED[j] = ' ';
                NAME_PADDED[maxNameLength] = 0;

                ImGui::Text(NAME_PADDED);
                ImGui::SameLine();
                ImGui::PushItemWidth(150.0f);
                ImGui::InputFloat(("##" + std::string(kernel::VAR_NAMES[i])).c_str(), &(kernel::VAR_VALUES[i]), 0.0f, 0.0f, "%f");
                ImGui::PopItemWidth();

                ImGui::SameLine();
                bool isRanging = kernel::VAR_RANGING[i];
                if (ImGui::Checkbox(("##RANGING_" + std::string(kernel::VAR_NAMES[i])).c_str(), &(isRanging)))
                {
                    kernel::VAR_RANGING[i] = !kernel::VAR_RANGING[i];
                }

                if (kernel::VAR_RANGING[i])
                {
                    ImGui::SameLine();
                    ImGui::PushItemWidth(150.0f);
                    ImGui::InputFloat(("##STEP_" + std::string(kernel::VAR_NAMES[i])).c_str(), &(kernel::VAR_STEPS[i]), 0.0f, 0.0f, "%f");

                    ImGui::SameLine();
                    ImGui::InputFloat(("##MAX_" + std::string(kernel::VAR_NAMES[i])).c_str(), &(kernel::VAR_MAX[i]), 0.0f, 0.0f, "%f");
                    ImGui::PopItemWidth();
                }
            }

            ImGui::SeparatorText("Parameters");

            for (int i = 0; i < kernel::PARAM_COUNT; i++)
            {
                NAME_PADDED = new char[maxNameLength + 1];
                strcpy(NAME_PADDED, kernel::PARAM_NAMES[i]);
                for (int j = (int)strlen(kernel::PARAM_NAMES[i]); j < maxNameLength; j++)
                    NAME_PADDED[j] = ' ';
                NAME_PADDED[maxNameLength] = 0;

                ImGui::Text(NAME_PADDED);
                ImGui::SameLine();
                ImGui::PushItemWidth(150.0f);
                ImGui::InputFloat(("##" + std::string(kernel::PARAM_NAMES[i])).c_str(), &(kernel::PARAM_VALUES[i]), 0.0f, 0.0f, "%f");
                ImGui::PopItemWidth();

                ImGui::SameLine();
                bool isRanging = kernel::PARAM_RANGING[i];
                if (ImGui::Checkbox(("##RANGING_" + std::string(kernel::PARAM_NAMES[i])).c_str(), &(isRanging)))
                {
                    kernel::PARAM_RANGING[i] = !kernel::PARAM_RANGING[i];
                }

                if (kernel::PARAM_RANGING[i])
                {
                    ImGui::SameLine();
                    ImGui::PushItemWidth(150.0f);
                    ImGui::InputFloat(("##STEP_" + std::string(kernel::PARAM_NAMES[i])).c_str(), &(kernel::PARAM_STEPS[i]), 0.0f, 0.0f, "%f");

                    ImGui::SameLine();
                    ImGui::InputFloat(("##MAX_" + std::string(kernel::PARAM_NAMES[i])).c_str(), &(kernel::PARAM_MAX[i]), 0.0f, 0.0f, "%f");
                    ImGui::PopItemWidth();
                }
            }

            // Modelling

            ImGui::SeparatorText("Modelling");
            ImGui::PushItemWidth(200.0f);
            ImGui::InputInt("Steps", &(kernel::steps), 1, 1000);
            ImGui::InputFloat("Step size", &(kernel::stepSize), 0.0f, 0.0f, "%f");
            ImGui::PopItemWidth();

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

            if (ImGui::Button("Compute"))
            {
                if (computedData) delete[] (float*)computedData;
                compute(&computedData, &rangingData);
                showComputedPlot = true;
                computedSteps = kernel::steps;
                if (dataBuffer) delete[] dataBuffer;
                dataBuffer = new float[(computedSteps + 1) * kernel::VAR_COUNT];

                autofitAfterComputing = true;
            }

            // Ranging

            variation = 0;
            stride = 1;

            if (rangingData.rangingCount > 0)
            {
                ImGui::SeparatorText("Ranging");

                for (int r = 0; r < rangingData.rangingCount; r++)
                {
                    float currentValue = rangingData.min[r] + rangingData.step[r] * rangingData.currentStep[r];
                    rangingData.currentValue[r] = currentValue;

                    ImGui::PushItemWidth(150.0f);
                    ImGui::DragInt(("##" + rangingData.names[r] + "_ranging").c_str(), &(rangingData.currentStep[r]), 1.0f, 0, rangingData.stepCount[r]-1);
                    ImGui::SameLine();
                    ImGui::Text((rangingData.names[r] + " = " + std::to_string(currentValue)).c_str());
                    ImGui::PopItemWidth();
                }

                for (int r = rangingData.rangingCount - 1; r >= 0; r--)
                {
                    variation += rangingData.currentStep[r] * stride;
                    stride *= rangingData.stepCount[r];
                }
            }

            // Graph Builder

            ImGui::SeparatorText("Graph Builder");

            ImGui::PushItemWidth(300.0f);
            ImGui::InputText("##Plot name input", plotNameBuffer, 64, ImGuiInputTextFlags_None);
            ImGui::PopItemWidth();

            std::string variablexyz[] = { "x", "y", "z" };

            ImGui::PushItemWidth(150.0f);
            for (int sv = 0; sv < 3; sv++)
            {
                ImGui::Text(("Variable " + variablexyz[sv]).c_str());
                ImGui::SameLine();
                if (ImGui::BeginCombo(("##Plot builder var " + std::to_string(sv + 1)).c_str(), selectedPlotVars[sv] > -1 ? kernel::VAR_NAMES[selectedPlotVars[sv]] : "-"))
                {
                    for (int v = (sv > 0 ? -1 : 0); v < kernel::VAR_COUNT; v++)
                    {
                        bool isSelected = selectedPlotVars[sv] == v;
                        ImGuiSelectableFlags selectableFlags = 0;

                        if (v == -1)
                        {
                            if (sv == 0 && (selectedPlotVars[1] > -1 || selectedPlotVars[2] > -1)) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (sv == 1 && (selectedPlotVars[2] > -1)) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (ImGui::Selectable("-", isSelected, selectableFlags)) selectedPlotVars[sv] = -1;
                        }
                        else
                        {
                            if (sv == 1 && selectedPlotVars[0] == -1) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (sv == 2 && selectedPlotVars[1] == -1) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (v == selectedPlotVars[(sv + 1) % 3] || v == selectedPlotVars[(sv + 2) % 3]) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (ImGui::Selectable(v > -1 ? kernel::VAR_NAMES[v] : "-", isSelected, selectableFlags)) selectedPlotVars[sv] = v;
                        }
                    }
                    ImGui::EndCombo();
                }
            }
            ImGui::PopItemWidth();            

            if (ImGui::Button("Create graph"))
            {
                plotWindows.push_back(PlotWindow(uniqueIds++, plotNameBuffer, true));
            }

            ImGui::End();
        }

        // PLOT WINDOWS //

        for (int w = 0; w < plotWindows.size(); w++)
        {
            PlotWindow* window = &(plotWindows[w]);
            if (!window->active)
            {
                plotWindows.erase(plotWindows.begin() + w);
                w--;
                continue;
            }

            style.WindowMenuButtonPosition = ImGuiDir_None;
            ImGui::Begin((window->name + std::to_string(window->id)).c_str(), &(window->active));

            if (window->yaw >= 360.0f) window->yaw -= 360.0f;
            if (window->yaw < 0.0f) window->yaw += 360.0f;

            if (window->pitch > 90.0f) window->pitch = 90.0f;
            if (window->pitch < -90.0f) window->pitch = -90.0f;

            ImGui::DragFloat("Yaw", &(window->yaw), 1.0f);
            ImGui::DragFloat("Pitch", &(window->pitch), 1.0f);

            ImGui::DragFloat("x offset", &(window->xOffset), 1.0f);
            ImGui::DragFloat("y offset", &(window->yOffset), 1.0f);
            ImGui::DragFloat("z offset", &(window->zOffset), 1.0f);

            ImPlotAxisFlags axisFlags = (autofitAfterComputing ? ImPlotAxisFlags_AutoFit : 0);
            ImPlot::BeginPlot(window->name.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, axisFlags, axisFlags);
            ImPlotPlot* plot = ImPlot::GetPlot(window->name.c_str());

            float plotRangeSize = (float)plot->Axes[ImAxis_X1].Range.Max - (float)plot->Axes[ImAxis_X1].Range.Min;

            plot->is3d = true;
            plot->deltax = &(window->yaw);
            plot->deltay = &(window->pitch);

            if (showComputedPlot)
            {
                int variationSize = kernel::VAR_COUNT * (computedSteps + 1);
                void* computedVariation = (float*)computedData + (variationSize * variation);
                memcpy(dataBuffer, computedVariation, variationSize * sizeof(float));

                populateAxisBuffer((float*)axisBuffer, plotRangeSize / 10, plotRangeSize / 10, plotRangeSize / 10);
                rotateOffsetBuffer((float*)axisBuffer, 6, 0, 1, 2, plotWindows[w].pitch, plotWindows[w].yaw, 0, 0, 0);

                if (computedData != nullptr) rotateOffsetBuffer((float*)dataBuffer, computedSteps + 1, 0, 1, 2,
                    plotWindows[w].pitch, plotWindows[w].yaw, plotWindows[w].xOffset, plotWindows[w].yOffset, plotWindows[w].zOffset);


                ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                ImPlot::PlotLine(window->name.c_str(), &(((float*)dataBuffer)[0]), &(((float*)dataBuffer)[1]), computedSteps + 1, 0, 0, sizeof(float) * 3);

                // Axis
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                ImPlot::PlotLine(window->name.c_str(), &(((float*)axisBuffer)[0]), &(((float*)axisBuffer)[1]), 2, 0, 0, sizeof(float) * 3);
                ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
                ImPlot::PlotLine(window->name.c_str(), &(((float*)axisBuffer)[6]), &(((float*)axisBuffer)[7]), 2, 0, 0, sizeof(float) * 3);
                ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.0f, 1.0f, 1.0f));
                ImPlot::PlotLine(window->name.c_str(), &(((float*)axisBuffer)[12]), &(((float*)axisBuffer)[13]), 2, 0, 0, sizeof(float) * 3);
            }
            ImPlot::EndPlot();

            ImGui::End();
        }

        autofitAfterComputing = false;

        // Rendering
        ImGui::Render();
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        // Present
        HRESULT hr = g_pSwapChain->Present(1, 0);   // Present with vsync
        //HRESULT hr = g_pSwapChain->Present(0, 0); // Present without vsync
        g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);
    }

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    if (computedData != nullptr) delete[] computedData;
    if (dataBuffer != nullptr) delete[] dataBuffer;
    if (axisBuffer != nullptr) delete[] axisBuffer;

    return 0;
}

// Helper functions
bool CreateDeviceD3D(HWND hWnd)
{
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    //createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED) // Try high-performance WARP software driver if hardware is not available.
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0 // From Windows SDK 8.1+ headers
#endif

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
            return 0;
        g_ResizeWidth = (UINT)LOWORD(lParam); // Queue resize
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    case WM_DPICHANGED:
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DpiEnableScaleViewports)
        {
            //const int dpi = HIWORD(wParam);
            //printf("WM_DPICHANGED to %d (%.0f%%)\n", dpi, (float)dpi / 96.0f * 100.0f);
            const RECT* suggested_rect = (RECT*)lParam;
            ::SetWindowPos(hWnd, nullptr, suggested_rect->left, suggested_rect->top, suggested_rect->right - suggested_rect->left, suggested_rect->bottom - suggested_rect->top, SWP_NOZORDER | SWP_NOACTIVATE);
        }
        break;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
