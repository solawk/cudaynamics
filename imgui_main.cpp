#include "imgui_main.h"

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

std::vector<PlotWindow> plotWindows;
int uniqueIds = 0; // Unique window IDs

void* computedData[2] = { nullptr, nullptr }; // For double buffering the computations, aren't null when computed
atomic_bool computedDataReady[2] = { false, false };
int playedBufferIndex = 0; // Buffer currently shown
int bufferToFillIndex = 0; // Buffer to send computations to

void* mapBuffers = nullptr;

void* dataBuffer = nullptr; // One variation local buffer
void* particleBuffer = nullptr; // One step local buffer
float* valuesOverride = nullptr; // For transferring end variable values to the next buffer
void* axisBuffer = new float[3 * 6] {}; // 3 axis, 6 points
void* gridBuffer = new float[10 * 5 * 3 * 2] {};
int computedSteps = 0; // Step count for the current computation
bool autofitAfterComputing = false; // Temporary flag to autofit computed data
PostRanging rangingData[2]; // Data about variation variables and parameters (1 per buffer for stability)
int currentTotalVariations = 0; // Current amount of variations, so we can compare and safely hot-swap the parameter values
bool executedOnLaunch = false; // Temporary flag to execute computations on launch if needed

float tempParamMin[kernel::PARAM_COUNT];
float tempParamMax[kernel::PARAM_COUNT];
float tempParamStep[kernel::PARAM_COUNT];

bool enabledParticles = true; // Particles mode
bool playingParticles = false; // Playing animation
float particleSpeed = 500.0f; // Steps per second
float particlePhase = 0.0f; // Animation frame cooldown
int particleStep = 0; // Current step of the computations to show
bool continuousComputingEnabled = false; // Continuously compute next batch of steps via double buffering

// Plot graph settings
bool markerSettingsWindowEnabled = true;
float markerSize = 1.0f;
float markerOutlineSize = 0.0f;
ImVec4 markerColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
ImPlotMarker markerShape = ImPlotMarker_Circle;
float gridAlpha = 0.15f;

// Temporary variables
int variation = 0;
int stride = 1;
float frameTime; // In seconds
float timeElapsed = 0.0f; // Ditto

// Colors
ImVec4 disabledColor = ImVec4(0.137f * 0.5f, 0.271f * 0.5f, 0.427f * 0.5f, 1.0f);
ImVec4 disabledTextColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);

std::future<int> computationFutures[2];

bool rangingWindowEnabled = true;
bool graphBuilderWindowEnabled = true;

void deleteBothBuffers()
{
    if (computedData[0] != nullptr) { delete[] computedData[0]; computedData[0] = nullptr; }
    if (computedData[1] != nullptr) { delete[] computedData[1]; computedData[1] = nullptr; }

    computedDataReady[0] = false;
    computedDataReady[1] = false;

    playedBufferIndex = 0;
    bufferToFillIndex = 0;
}

void resetOverrideBuffer(int totalVariations)
{
    if (valuesOverride) delete[] valuesOverride;
    valuesOverride = new float[kernel::VAR_COUNT * totalVariations];
}

void resetTempBuffers(int totalVariations)
{
    if (dataBuffer) delete[] dataBuffer;
    dataBuffer = new float[(computedSteps + 1) * kernel::VAR_COUNT];

    if (particleBuffer) delete[] particleBuffer;
    particleBuffer = new float[totalVariations * kernel::VAR_COUNT];
}

void computing();

void printResults(vector<Point>& points, int num_points)
{
    int i = 0;
    printf("Number of points: %u\n"
        " x     y     z     cluster_id\n"
        "-----------------------------\n"
        , num_points);
    while (i < num_points)
    {
        printf("%5.2lf %5.2lf %5.2lf: %d\n",
            points[i].x,
            points[i].y, points[i].z,
            points[i].clusterID);
        ++i;
    }
}

int asyncComputation(void** dest, PostRanging* rangingData)
{
    computedDataReady[bufferToFillIndex] = false;

    bool isFirstBatch = computedData[1 - bufferToFillIndex] == nullptr; // Is another buffer null, only true when computing for the first time
    if (isFirstBatch) rangingData->clear();

    printf("is first batch %i, total variations %i\n", isFirstBatch, rangingData->totalVariations);

    int computationResult = compute(dest, &mapBuffers, isFirstBatch ? nullptr : (float*)(computedData[1 - bufferToFillIndex]), rangingData);

    computedSteps = kernel::steps;

    if (isFirstBatch)
    {
        autofitAfterComputing = true;
        resetTempBuffers(rangingData->totalVariations);
        resetOverrideBuffer(rangingData->totalVariations);
    }

    // WIP dbscan
    if (0)
    {
        vector<Point> dbPoints(rangingData->totalVariations);
        int variationSize = kernel::VAR_COUNT * (computedSteps + 1);
        for (int i = 0; i < rangingData->totalVariations; i++)
        {
            dbPoints[i].x = ((float*)(*dest))[(variationSize * i) + (kernel::VAR_COUNT * (computedSteps - 1)) + kernel::sin_x0];
            dbPoints[i].y = ((float*)(*dest))[(variationSize * i) + (kernel::VAR_COUNT * (computedSteps - 1)) + kernel::x1];
            dbPoints[i].z = ((float*)(*dest))[(variationSize * i) + (kernel::VAR_COUNT * (computedSteps - 1)) + kernel::x2];
        }
        DBSCAN ds(4, 10000.0f * 10000.0f, dbPoints);
        ds.run();
        printResults(ds.m_points, ds.getTotalPointSize());
    }

    computedDataReady[bufferToFillIndex] = true;

    if (continuousComputingEnabled) bufferToFillIndex = 1 - bufferToFillIndex;
    if (continuousComputingEnabled && bufferToFillIndex != playedBufferIndex)
    {
        computing();
    }

    return computationResult;
}

void computing()
{
    computationFutures[bufferToFillIndex] = std::async(asyncComputation, &(computedData[bufferToFillIndex]), &(rangingData[bufferToFillIndex]));
}

// Windows configuration saving and loading
void saveWindows()
{
    ofstream configFileStream((std::string(kernel::name) + ".config").c_str(), ios::out);

    for (PlotWindow w : plotWindows)
    {
        string exportString = w.ExportAsString();
        configFileStream.write(exportString.c_str(), exportString.length());
    }

    configFileStream.close();
}
void loadWindows()
{
    ifstream configFileStream((std::string(kernel::name) + ".config").c_str(), ios::in);

    for (std::string line; getline(configFileStream, line); )
    {
        PlotWindow plotWindow = PlotWindow(uniqueIds++);
        plotWindow.ImportAsString(line);

        plotWindows.push_back(plotWindow);
    }

    configFileStream.close();
}

// Main code
int imgui_main(int, char**)
{
    // Create application window
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), LoadIcon(wc.hInstance, MAKEINTRESOURCE(IDI_ICON1)), nullptr, nullptr, nullptr, L"CUDAynamics", LoadIcon(wc.hInstance, MAKEINTRESOURCE(IDI_ICON1)) };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"CUDAynamics", WS_OVERLAPPEDWINDOW, 100, 100, 400, 100, nullptr, nullptr, wc.hInstance, nullptr);

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

    //io.Fonts->AddFontDefault();
    io.Fonts->AddFontFromFileTTF("UbuntuMono-R.ttf", 24.0f);
    //IM_ASSERT(font != nullptr);

    // Main loop
    bool work = true;

    char* plotNameBuffer = new char[64]();
    strcpy_s(plotNameBuffer, 5, "Plot");

    PlotType plotType = Series;
    int selectedPlotVars[3]; selectedPlotVars[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVars[i] = -1;
    set<int> selectedPlotVarsSet;
    int selectedPlotMap = 0;

    memcpy(tempParamMin, kernel::PARAM_VALUES, sizeof(float) * kernel::PARAM_COUNT);
    memcpy(tempParamMax, kernel::PARAM_MAX, sizeof(float) * kernel::PARAM_COUNT);
    memcpy(tempParamStep, kernel::PARAM_STEPS, sizeof(float) * kernel::PARAM_COUNT);

    try
    {
        loadWindows();
    }
    catch (exception e)
    {
        printf(e.what());
    }

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

        timeElapsed += frameTime;
        float breath = (cosf(timeElapsed * 6.0f) + 1.0f) / 2.0f;
        float buttonBreathMult = 1.2f + breath * 0.8f;

        if (particleStep > computedSteps) particleStep = computedSteps;

        // MAIN WINDOW
        {
            style.WindowMenuButtonPosition = ImGuiDir_Left;
            ImGui::Begin("CUDAynamics", &work);
            ImGui::Text(kernel::name);

            // Parameters & Variables

            ImGuiSliderFlags dragFlag = !playingParticles ? 0 : ImGuiSliderFlags_ReadOnly;

            string namePadded;
            int maxNameLength = 0;

            for (int i = 0; i < kernel::PARAM_COUNT; i++) if (strlen(kernel::PARAM_NAMES[i]) > maxNameLength) maxNameLength = (int)strlen(kernel::PARAM_NAMES[i]);
            for (int i = 0; i < kernel::VAR_COUNT; i++) if (strlen(kernel::VAR_NAMES[i]) > maxNameLength) maxNameLength = (int)strlen(kernel::VAR_NAMES[i]);          

            ImGui::SeparatorText("Variables");

            for (int i = 0; i < kernel::VAR_COUNT; i++)
            {
                namePadded = kernel::VAR_NAMES[i];
                for (int j = (int)strlen(kernel::VAR_NAMES[i]); j < maxNameLength; j++)
                    namePadded += ' ';

                ImGui::Text(namePadded.c_str());


                if (playingParticles) ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
                ImGui::SameLine();
                ImGui::PushItemWidth(150.0f);
                ImGui::DragFloat(("##" + std::string(kernel::VAR_NAMES[i])).c_str(), &(kernel::VAR_VALUES[i]), 1.0f, 0.0f, 0.0f, "%f", dragFlag);
                ImGui::PopItemWidth();

                ImGui::SameLine();
                bool isRanging = kernel::VAR_RANGING[i];
                if (ImGui::Checkbox(("##RANGING_" + std::string(kernel::VAR_NAMES[i])).c_str(), &(isRanging)) && !playingParticles)
                {
                    kernel::VAR_RANGING[i] = !kernel::VAR_RANGING[i];
                }

                if (kernel::VAR_RANGING[i])
                {
                    ImGui::SameLine();
                    ImGui::PushItemWidth(150.0f);
                    ImGui::DragFloat(("##STEP_" + std::string(kernel::VAR_NAMES[i])).c_str(), &(kernel::VAR_STEPS[i]), 1.0f, 0.0f, 0.0f, "%f", dragFlag);

                    ImGui::SameLine();
                    ImGui::DragFloat(("##MAX_" + std::string(kernel::VAR_NAMES[i])).c_str(), &(kernel::VAR_MAX[i]), 1.0f, 0.0f, 0.0f, "%f", dragFlag);
                    ImGui::PopItemWidth();

                    ImGui::SameLine();
                    ImGui::Text((std::to_string(calculateStepCount(kernel::VAR_VALUES[i], kernel::VAR_MAX[i], kernel::VAR_STEPS[i])) + " steps").c_str());
                }
                if (playingParticles) ImGui::PopStyleColor();
            }

            ImGui::SeparatorText("Parameters");
            bool anyChanged = false;

            for (int i = 0; i < kernel::PARAM_COUNT; i++)
            {
                namePadded = kernel::PARAM_NAMES[i];
                for (int j = (int)strlen(kernel::PARAM_NAMES[i]); j < maxNameLength; j++)
                    namePadded += ' ';

                ImGui::Text(namePadded.c_str());

                float* paramMin = !playingParticles ? &(kernel::PARAM_VALUES[i]) : &(tempParamMin[i]);
                float* paramMax = !playingParticles ? &(kernel::PARAM_MAX[i]) : &(tempParamMax[i]);
                float* paramStep = !playingParticles ? &(kernel::PARAM_STEPS[i]) : &(tempParamStep[i]);
                bool tempChanged = (kernel::PARAM_VALUES[i] != tempParamMin[i]) || (kernel::PARAM_MAX[i] != tempParamMax[i]) || (kernel::PARAM_STEPS[i] != tempParamStep[i]);
                if (tempChanged) anyChanged = true;

                //if (playingParticles) ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
                ImGui::SameLine();
                ImGui::PushItemWidth(150.0f);
                ImGui::DragFloat(("##" + std::string(kernel::PARAM_NAMES[i])).c_str(), paramMin, 1.0f, 0.0f, 0.0f, "%f", 0);
                ImGui::PopItemWidth();

                ImGui::SameLine();
                bool isRanging = kernel::PARAM_RANGING[i];
                if (ImGui::Checkbox(("##RANGING_" + std::string(kernel::PARAM_NAMES[i])).c_str(), &(isRanging)) && !playingParticles)
                {
                    kernel::PARAM_RANGING[i] = !kernel::PARAM_RANGING[i];
                }

                if (kernel::PARAM_RANGING[i])
                {
                    ImGui::SameLine();
                    ImGui::PushItemWidth(150.0f);
                    ImGui::DragFloat(("##STEP_" + std::string(kernel::PARAM_NAMES[i])).c_str(), paramStep, 1.0f, 0.0f, 0.0f, "%f", 0);

                    ImGui::SameLine();
                    ImGui::DragFloat(("##MAX_" + std::string(kernel::PARAM_NAMES[i])).c_str(), paramMax, 1.0f, 0.0f, 0.0f, "%f", 0);
                    ImGui::PopItemWidth();

                    int stepCount = !playingParticles ? calculateStepCount(kernel::PARAM_VALUES[i], kernel::PARAM_MAX[i], kernel::PARAM_STEPS[i]) :
                                                        calculateStepCount(tempParamMin[i], tempParamMax[i], tempParamStep[i]);
                    if (stepCount > 0)
                    {
                        ImGui::SameLine();
                        ImGui::Text((std::to_string(stepCount) + " steps").c_str());
                    }
                }
                //if (playingParticles) ImGui::PopStyleColor();
            }

            if (playingParticles && anyChanged)
            {
                int tempTotalVariations = 1;
                for (int v = 0; v < kernel::VAR_COUNT; v++) if (kernel::VAR_RANGING[v]) tempTotalVariations *= (calculateStepCount(kernel::VAR_VALUES[v], kernel::VAR_MAX[v], kernel::VAR_STEPS[v]));
                for (int p = 0; p < kernel::PARAM_COUNT; p++) if (kernel::PARAM_RANGING[p])  tempTotalVariations *= (calculateStepCount(tempParamMin[p], tempParamMax[p], tempParamStep[p]));

                if (tempTotalVariations != rangingData[playedBufferIndex].totalVariations)
                {
                    ImGui::Text((std::string("Cannot hot-swap (now ") + std::to_string(tempTotalVariations)
                        + std::string(" variations, was ") + std::to_string(rangingData[playedBufferIndex].totalVariations) + std::string(" variations)")).c_str());
                }
                else
                {
                    if (ImGui::Button("Apply hot-swap"))
                    {
                        memcpy(kernel::PARAM_VALUES, tempParamMin, sizeof(float) * kernel::PARAM_COUNT);
                        memcpy(kernel::PARAM_MAX, tempParamMax, sizeof(float) * kernel::PARAM_COUNT);
                        memcpy(kernel::PARAM_STEPS, tempParamStep, sizeof(float) * kernel::PARAM_COUNT);
                    }
                }
            }

            // Modelling

            ImGui::SeparatorText("Simulation");
            ImGui::PushItemWidth(200.0f);
            ImGui::InputInt("Steps", &(kernel::steps), 1, 1000);
            ImGui::InputFloat("Step size", &(kernel::stepSize), 0.0f, 0.0f, "%f");
            ImGui::PopItemWidth();
            
            bool tempEnabledParticles = enabledParticles;
            if (ImGui::Checkbox("Particles mode", &(tempEnabledParticles)))
            {
                enabledParticles = !enabledParticles;
            }

            if (tempEnabledParticles)
            {
                ImGui::SameLine();
                ImGui::PushItemWidth(200.0f);
                ImGui::DragFloat("Animation speed", &(particleSpeed), 1.0f);
                if (particleSpeed < 0.0f) particleSpeed = 0.0f;
                ImGui::PopItemWidth();
                ImGui::DragInt("Animation step", &(particleStep), 1.0f, 0, kernel::steps);

                if (ImGui::Button("Reset to step 0"))
                {
                    particleStep = 0;
                }

                bool tempPlayingParticles = playingParticles;
                if (ImGui::Checkbox("Play", &(tempPlayingParticles)))
                {
                    if (computedDataReady[0])
                    {
                        playingParticles = !playingParticles;

                        memcpy(tempParamMin, kernel::PARAM_VALUES, sizeof(float) * kernel::PARAM_COUNT);
                        memcpy(tempParamMax, kernel::PARAM_MAX, sizeof(float) * kernel::PARAM_COUNT);
                        memcpy(tempParamStep, kernel::PARAM_STEPS, sizeof(float) * kernel::PARAM_COUNT);
                    }
                }

                bool tempContinuous = continuousComputingEnabled;
                if (ImGui::Checkbox("Continuous computing", &(tempContinuous)))
                {
                    continuousComputingEnabled = !continuousComputingEnabled;
                    
                    deleteBothBuffers();
                    playingParticles = false;
                    particleStep = 0;
                }

                bool tempMarkerSettings = markerSettingsWindowEnabled;
                if (ImGui::Checkbox("Plot graph settings", &(tempMarkerSettings)))
                {
                    markerSettingsWindowEnabled = !markerSettingsWindowEnabled;
                }
            }

            frameTime = 1.0f / io.Framerate;
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            //ImGui::Text("BufferToFill %i PlayedBuffer %i", bufferToFillIndex, playedBufferIndex);

            if (playingParticles && enabledParticles)
            {
                particlePhase += frameTime * particleSpeed;
                int passedSteps = (int)floor(particlePhase);
                particlePhase -= (float)passedSteps;

                particleStep += passedSteps;
                if (particleStep > kernel::steps) // Reached the end of animation
                {
                    if (continuousComputingEnabled)
                    {
                        // Starting from another buffer

                        if (computedDataReady[1 - playedBufferIndex])
                        {
                            playedBufferIndex = 1 - playedBufferIndex;
                            particleStep = 0;
                            printf("Switch occured and starting playing %i\n", playedBufferIndex);
                            computing();
                        }
                        else
                        {
                            //printf("Stalling!\n");
                            particleStep = kernel::steps;

                            ImGui::Text("Stalling!");
                        }
                    }
                    else
                    {
                        // Stopping
                        particleStep = kernel::steps;
                        playingParticles = false;
                    }                   
                }
            }

            // default button color is 0.137 0.271 0.427
            bool noComputedData = computedData[0] == nullptr;
            if (noComputedData) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.137f * buttonBreathMult, 0.271f * buttonBreathMult, 0.427f * buttonBreathMult, 1.0f));
            if (ImGui::Button("= COMPUTE =") || (kernel::executeOnLaunch && !executedOnLaunch))
            {
                executedOnLaunch = true;
                bufferToFillIndex = 0;
                playedBufferIndex = 0;
                deleteBothBuffers();

                computing();
            }
            if (noComputedData) ImGui::PopStyleColor();

            ImGui::End();

            // Ranging

            variation = 0;
            stride = 1;

            if (rangingData[playedBufferIndex].rangingCount > 0 && computedDataReady[playedBufferIndex])
            {
                for (int r = rangingData[playedBufferIndex].rangingCount - 1; r >= 0; r--)
                {
                    variation += rangingData[playedBufferIndex].currentStep[r] * stride;
                    stride *= rangingData[playedBufferIndex].stepCount[r];
                }

                currentTotalVariations = rangingData->totalVariations;

                if (rangingWindowEnabled)
                {
                    ImGui::Begin("Ranging", &rangingWindowEnabled);

                    for (int r = 0; r < rangingData[playedBufferIndex].rangingCount; r++)
                    {
                        float currentValue = rangingData[playedBufferIndex].min[r] + rangingData[playedBufferIndex].step[r] * rangingData[playedBufferIndex].currentStep[r];
                        rangingData[playedBufferIndex].currentValue[r] = currentValue;

                        ImGui::PushItemWidth(150.0f);
                        ImGui::DragInt(("##" + rangingData[playedBufferIndex].names[r] + "_ranging").c_str(), &(rangingData[playedBufferIndex].currentStep[r]), 1.0f, 0, rangingData[playedBufferIndex].stepCount[r] - 1);
                        ImGui::SameLine();
                        ImGui::Text((rangingData[playedBufferIndex].names[r] + " = " + std::to_string(currentValue)).c_str());
                        ImGui::PopItemWidth();
                    }

                    ImGui::Text(("Current variations: " + std::to_string(currentTotalVariations)).c_str());

                    ImGui::End();
                }
            }

            //ImGui::Text((std::string("Ready 0 ") + std::to_string(computedDataReady[0])).c_str());
            //ImGui::Text((std::string("Ready 1 ") + std::to_string(computedDataReady[1])).c_str());

            // Graph Builder

            if (graphBuilderWindowEnabled)
            {
                ImGui::Begin("Graph Builder", &graphBuilderWindowEnabled);

                //ImGui::PushItemWidth(300.0f);
                //ImGui::InputText("##Plot name input", plotNameBuffer, 64, ImGuiInputTextFlags_None);
                //ImGui::PopItemWidth();

                // Type
                std::string plottypes[] = { "Time series", "Phase diagram", "Orbit diagram", "Heatmap" };
                ImGui::Text("Plot type ");
                ImGui::SameLine();
                ImGui::PushItemWidth(250.0f);
                if (ImGui::BeginCombo("##Plot type", (plottypes[plotType]).c_str()))
                {
                    for (int t = 0; t < PlotType_COUNT; t++)
                    {
                        bool isSelected = plotType == t;
                        ImGuiSelectableFlags selectableFlags = 0;
                        if (ImGui::Selectable(plottypes[t].c_str(), isSelected, selectableFlags)) plotType = (PlotType)t;
                    }

                    ImGui::EndCombo();
                }
                ImGui::PopItemWidth();

                std::string variablexyz[] = { "x", "y", "z" };

                switch (plotType)
                {
                case Series:

                    // Variable adding combo

                    ImGui::Text("Add variable");
                    ImGui::SameLine();
                    if (ImGui::BeginCombo("##Add variable combo", " ", ImGuiComboFlags_NoPreview))
                    {
                        for (int v = 0; v < kernel::VAR_COUNT; v++)
                        {
                            bool isSelected = selectedPlotVarsSet.find(v) != selectedPlotVarsSet.end();
                            ImGuiSelectableFlags selectableFlags = 0;

                            if (isSelected) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (ImGui::Selectable(kernel::VAR_NAMES[v])) selectedPlotVarsSet.insert(v);
                        }

                        ImGui::EndCombo();
                    }

                    // Variable list

                    for (const int& v : selectedPlotVarsSet)
                    {
                        if (ImGui::Button(("x##" + std::to_string(v)).c_str()))
                        {
                            selectedPlotVarsSet.erase(v);
                        }
                        ImGui::SameLine();
                        ImGui::Text(("- " + std::string(kernel::VAR_NAMES[v])).c_str());
                    }

                    break;

                case Phase:
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
                    break;

                case Orbit:

                    break;

                case Heatmap:
                    if (kernel::MAP_COUNT > 0)
                    {
                        ImGui::PushItemWidth(150.0f);

                        ImGui::Text("Index");
                        ImGui::SameLine();
                        if (ImGui::BeginCombo("##Plot builder map index selection", kernel::MAP_NAMES[selectedPlotMap]))
                        {
                            for (int m = 0; m < kernel::MAP_COUNT; m++)
                            {
                                bool isSelected = selectedPlotMap == m;
                                ImGuiSelectableFlags selectableFlags = 0;

                                if (selectedPlotMap == m) selectableFlags = ImGuiSelectableFlags_Disabled;
                                if (ImGui::Selectable(kernel::MAP_NAMES[m], isSelected, selectableFlags)) selectedPlotMap = m;
                            }
                            ImGui::EndCombo();
                        }

                        ImGui::PopItemWidth();
                    }
                    break;
                }

                if (ImGui::Button("Create graph"))
                {
                    PlotWindow plotWindow = PlotWindow(uniqueIds++, plotNameBuffer, true);
                    plotWindow.type = plotType;

                    if (plotType == Series) plotWindow.AssignVariables(selectedPlotVarsSet);
                    if (plotType == Phase) plotWindow.AssignVariables(selectedPlotVars);
                    if (plotType == Heatmap) plotWindow.AssignVariables(selectedPlotMap);

                    plotWindows.push_back(plotWindow);
                }

                ImGui::End();
            }
        }

        // PLOT GRAPH SETTINGS WINDOW //

        if (markerSettingsWindowEnabled)
        {
            ImGui::Begin("Plot graph settings", &markerSettingsWindowEnabled);

            ImGui::DragFloat("Marker size", &markerSize, 0.1f);
            ImGui::DragFloat("Marker outline size", &markerOutlineSize, 0.1f);
            if (markerSize < 0.0f) markerSize = 0.0f;
            ImGui::ColorEdit4("Marker color", (float*)(&markerColor));

            std::string shapeNames[]{ "Circle", "Square", "Diamond", "Up", "Down", "Left", "Right", "Cross", "Plus", "Asterisk" };
            if (ImGui::BeginCombo("Marker shape", shapeNames[markerShape].c_str()))
            {
                for (ImPlotMarker i = 0; i < ImPlotMarker_COUNT; i++)
                {
                    bool isSelected = markerShape == i;
                    if (ImGui::Selectable(shapeNames[i].c_str(), isSelected)) markerShape = i;
                }
                ImGui::EndCombo();
            }

            ImGui::DragFloat("Grid alpha", &gridAlpha, 0.01f);

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
            std::string windowName = window->name + std::to_string(window->id);
            std::string plotName = windowName + "_plot";
            ImGui::Begin(windowName.c_str(), &(window->active));

            // Common variables
            ImPlotAxisFlags axisFlags = (autofitAfterComputing ? ImPlotAxisFlags_AutoFit : 0);
            ImPlotPlot* plot;
            bool is3d;
            int mapIndex;
            ImPlotColormap heatmapColorMap = ImPlotColormap_Jet;

            switch (window->type)
            {
            case Series:

                //printf("Begin series\n");
                if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle, axisFlags, axisFlags))
                {
                    plot = ImPlot::GetPlot(plotName.c_str());

                    plot->is3d = false;

                    if (computedDataReady[playedBufferIndex])
                    {
                        int variationSize = kernel::VAR_COUNT * (computedSteps + 1);

                        void* computedVariation = (float*)(computedData[playedBufferIndex]) + (variationSize * variation);
                        memcpy(dataBuffer, computedVariation, variationSize * sizeof(float));

                        //void PlotLine(const char* label_id, const T* values, int count, double xscale, double x0, ImPlotLineFlags flags, int offset, int stride)

                        for (int v = 0; v < window->variableCount; v++)
                        {
                            //ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                            ImPlot::PlotLine((std::string(kernel::VAR_NAMES[window->variables[v]]) + "##" + plotName + std::to_string(v)).c_str(),
                                &(((float*)dataBuffer)[window->variables[v]]), computedSteps + 1, 1.0f, 0.0f, ImPlotLineFlags_None, 0, sizeof(float) * kernel::VAR_COUNT);
                        }
                    }

                    //printf("End series\n");
                    ImPlot::EndPlot();
                }

                break;

            case Phase:
                // PHASE DIAGRAM
                is3d = window->variableCount == 3;

                if (is3d)
                {
                    if (window->rotation.x >= 360.0f) window->rotation.x -= 360.0f;
                    if (window->rotation.x < 0.0f) window->rotation.x += 360.0f;

                    if (window->rotation.y > 90.0f) window->rotation.y = 90.0f;
                    if (window->rotation.y < -90.0f) window->rotation.y = -90.0f;

                    if (window->scale.x < 0.0f) window->scale.x = 0.0f;
                    if (window->scale.y < 0.0f) window->scale.y = 0.0f;
                    if (window->scale.z < 0.0f) window->scale.z = 0.0f;

                    ImGui::DragFloat2("Rotation", (float*)&window->rotation, 1.0f);
                    ImGui::DragFloat3("Offset", (float*)&window->offset, 0.01f);
                    ImGui::DragFloat3("Scale", (float*)&window->scale, 0.01f);

                    axisFlags |= ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels;
                }

                //printf("Begin phase\n");
                if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, axisFlags, axisFlags))
                {
                    plot = ImPlot::GetPlot(plotName.c_str());

                    float plotRangeSize = (float)plot->Axes[ImAxis_X1].Range.Max - (float)plot->Axes[ImAxis_X1].Range.Min;

                    plot->is3d = is3d;
                    plot->deltax = &(window->rotation.x);
                    plot->deltay = &(window->rotation.y);

                    if (computedDataReady[playedBufferIndex])
                    {
                        int variationSize = kernel::VAR_COUNT * (computedSteps + 1);

                        populateAxisBuffer((float*)axisBuffer, plotRangeSize / 10, plotRangeSize / 10, plotRangeSize / 10);
                        if (is3d)
                        {
                            rotateOffsetBuffer((float*)axisBuffer, 6, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 0));
                        }

                        int xIndex = is3d ? 0 : window->variables[0];
                        int yIndex = is3d ? 1 : window->variables[1];

                        if (!enabledParticles) // Trajectory - one variation, all steps
                        {
                            void* computedVariation = (float*)(computedData[playedBufferIndex]) + (variationSize * variation);
                            memcpy(dataBuffer, computedVariation, variationSize * sizeof(float));

                            if (is3d)
                                rotateOffsetBuffer((float*)dataBuffer, computedSteps + 1, kernel::VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                    window->rotation.y, window->rotation.x, window->offset, window->scale);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)dataBuffer)[xIndex]), &(((float*)dataBuffer)[yIndex]), computedSteps + 1, 0, 0, sizeof(float) * kernel::VAR_COUNT);
                        }
                        else // Particles - all variations, one certain step
                        {
                            for (int v = 0; v < rangingData[playedBufferIndex].totalVariations; v++)
                            {
                                for (int var = 0; var < kernel::VAR_COUNT; var++)
                                    ((float*)particleBuffer)[v * kernel::VAR_COUNT + var] = ((float*)(computedData[playedBufferIndex]))[(variationSize * v) + (kernel::VAR_COUNT * particleStep) + var];
                            }

                            if (is3d)
                                rotateOffsetBuffer((float*)particleBuffer, rangingData[playedBufferIndex].totalVariations, kernel::VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                    window->rotation.y, window->rotation.x, window->offset, window->scale);

                            ImPlot::SetNextLineStyle(markerColor);
                            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, markerOutlineSize);
                            //ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
                            ImPlot::SetNextMarkerStyle(markerShape, markerSize);
                            ImPlot::PlotScatter(plotName.c_str(), &(((float*)particleBuffer)[xIndex]), &(((float*)particleBuffer)[yIndex]), rangingData[playedBufferIndex].totalVariations, 0, 0, sizeof(float) * kernel::VAR_COUNT);
                        }

                        // Axis
                        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                        ImPlot::PlotLine(plotName.c_str(), &(((float*)axisBuffer)[0]), &(((float*)axisBuffer)[1]), 2, 0, 0, sizeof(float) * 3);
                        ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
                        ImPlot::PlotLine(plotName.c_str(), &(((float*)axisBuffer)[6]), &(((float*)axisBuffer)[7]), 2, 0, 0, sizeof(float) * 3);

                        if (is3d)
                        {
                            ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.0f, 1.0f, 1.0f));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)axisBuffer)[12]), &(((float*)axisBuffer)[13]), 2, 0, 0, sizeof(float) * 3);
                        }

                        // Axis names
                        ImPlot::PushStyleColor(ImPlotCol_InlayText, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
                        ImPlot::PlotText(kernel::VAR_NAMES[window->variables[0]], ((float*)axisBuffer)[0], ((float*)axisBuffer)[1], ImVec2(0.0f, 0.0f));
                        ImPlot::PopStyleColor();
                        ImPlot::PushStyleColor(ImPlotCol_InlayText, ImVec4(0.2f, 1.0f, 0.2f, 1.0f));
                        ImPlot::PlotText(kernel::VAR_NAMES[window->variables[1]], ((float*)axisBuffer)[6], ((float*)axisBuffer)[7], ImVec2(0.0f, 0.0f));
                        ImPlot::PopStyleColor();

                        if (is3d)
                        {
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, ImVec4(0.2f, 0.2f, 1.0f, 1.0f));
                            ImPlot::PlotText(kernel::VAR_NAMES[window->variables[2]], ((float*)axisBuffer)[12], ((float*)axisBuffer)[13], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                        }

                        // Grid
                        if (is3d)
                        {
                            ImVec4 scale(plotRangeSize/* * window->xScale*/, plotRangeSize/* * window->yScale*/, plotRangeSize/* * window->zScale*/, 0);
                            ImVec4 scaleLog(floorf(log10f(scale.x)), floorf(log10f(scale.y)), floorf(log10f(scale.z)), 0);
                            ImVec4 scale0(powf(10, scaleLog.x - 1), powf(10, scaleLog.y - 1), powf(10, scaleLog.z - 1), 0);
                            ImVec4 scale1(powf(10, scaleLog.x), powf(10, scaleLog.y), powf(10, scaleLog.z), 0);
                            ImVec4 scaleInterp(log10f(scale.x) - scaleLog.x, log10f(scale.y) - scaleLog.y, log10f(scale.z) - scaleLog.z, 0);

                            ImVec4 alpha0((1.0f - scaleInterp.x) * gridAlpha, (1.0f - scaleInterp.y) * gridAlpha, (1.0f - scaleInterp.z) * gridAlpha, 0);
                            ImVec4 alpha1(scaleInterp.x * gridAlpha, scaleInterp.y * gridAlpha, scaleInterp.z * gridAlpha, 0);

                            // x
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha0.x));
                            populateGridBuffer((float*)gridBuffer);
                            rotateOffsetBuffer((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale0.x * window->scale.x, scale0.y * window->scale.y, scale0.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            //ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5, 0, 50 * 3, sizeof(float) * 3);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha1.x));
                            populateGridBuffer((float*)gridBuffer);
                            rotateOffsetBuffer((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale1.x * window->scale.x, scale1.y * window->scale.y, scale1.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            //ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5, 0, 50 * 3, sizeof(float) * 3);

                            // y
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha0.y));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            rotateOffsetBuffer((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale0.x * window->scale.x, scale0.y * window->scale.y, scale0.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            //ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5, 0, 50 * 3, sizeof(float) * 3);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha1.y));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            rotateOffsetBuffer((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale1.x * window->scale.x, scale1.y * window->scale.y, scale1.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            //ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5, 0, 50 * 3, sizeof(float) * 3);

                            // z
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha0.z));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            gridY2Z((float*)gridBuffer);
                            rotateOffsetBuffer((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale0.x * window->scale.x, scale0.y * window->scale.y, scale0.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            //ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5, 0, 50 * 3, sizeof(float) * 3);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha1.z));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            gridY2Z((float*)gridBuffer);
                            rotateOffsetBuffer((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale1.x * window->scale.x, scale1.y * window->scale.y, scale1.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            //ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5, 0, 50 * 3, sizeof(float) * 3);
                        }
                    }

                    // PHASE DIAGRAM END
                    //printf("End phase\n");
                    ImPlot::EndPlot();
                }
                break;

                case Heatmap:

                    //IMPLOT_TMP void PlotHeatmap(const char* label_id, const T* values, int rows, int cols, double scale_min=0, double scale_max=0, const char* label_fmt="%.1f", const ImPlotPoint& bounds_min=ImPlotPoint(0,0), const ImPlotPoint& bounds_max=ImPlotPoint(1,1), ImPlotHeatmapFlags flags=0);

                    if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend, axisFlags, axisFlags))
                    {
                        plot = ImPlot::GetPlot(plotName.c_str());
                        plot->is3d = false;

                        ImPlot::PushColormap(heatmapColorMap);
                        mapIndex = window->variables[0];
                        if (mapBuffers)
                        {
                            ImPlot::PlotHeatmap((std::string(kernel::VAR_NAMES[mapIndex]) + "##" + plotName + std::to_string(0)).c_str(),
                                ((float*)mapBuffers + mapIndex * rangingData->totalVariations),
                                kernel::VAR_STEP_COUNTS[kernel::MAP_X[mapIndex]], kernel::VAR_STEP_COUNTS[kernel::MAP_Y[mapIndex]],
                                0, 0, "", ImPlotPoint(0, 0), ImPlotPoint(kernel::VAR_STEP_COUNTS[kernel::MAP_X[mapIndex]], kernel::VAR_STEP_COUNTS[kernel::MAP_Y[mapIndex]]));
                        }
                        ImPlot::PopColormap();

                        //printf("End series\n");
                        ImPlot::EndPlot();
                    }

                    break;
            }
           

            ImGui::End();
        }

        autofitAfterComputing = false;

        // Rendering
        ImGui::Render();
        ImVec4 clear_color = ImVec4(1.0f, 1.0f, 1.0f, 1.00f);
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

    saveWindows();

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    if (computationFutures[0].valid()) computationFutures[0].wait();
    if (computationFutures[1].valid()) computationFutures[1].wait();
    deleteBothBuffers();
    if (dataBuffer != nullptr) delete[] dataBuffer;
    if (particleBuffer != nullptr) delete[] particleBuffer;
    if (valuesOverride != nullptr) delete[] valuesOverride;
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
