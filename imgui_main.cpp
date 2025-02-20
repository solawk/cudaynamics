#include "imgui_main.h"

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

std::vector<PlotWindow> plotWindows;
int uniqueIds = 0; // Unique window IDs

void** initialValues = nullptr; // Array of all initial values of variables and parameters, can be not regenerated at will (e.g., user wants the same random values)
// When changes are applied to initial values, nullified automatically
// Not regenerated when hot-swapping (since there is no point to it anyway)
// Is generated before every launch, unless disabled

void* computedData[2] = { nullptr, nullptr }; // For double buffering the computations, aren't null when computed
atomic_bool computedDataReady[2] = { false, false };
int playedBufferIndex = 0; // Buffer currently shown
int bufferToFillIndex = 0; // Buffer to send computations to
void* mapBuffers[2] = { nullptr, nullptr };

InputValuesBuffer<numb> varNew;
bool autoLoadNewParams = false;
InputValuesBuffer<numb> paramNew;
int stepsNew = 0;

void* dataBuffer = nullptr; // One variation local buffer
void* particleBuffer = nullptr; // One step local buffer
numb* valuesOverride = nullptr; // For transferring end variable values to the next buffer

void* axisBuffer = new numb[3 * 2 * 3] {}; // 3 axis, 2 points
void* rulerBuffer = new numb[51 * 3] {}; // 1 axis, 5 * 10 + 1 points
void* gridBuffer = new numb[10 * 5 * 3 * 2] {};

int computedSteps = 0; // Step count for the current computation
bool autofitAfterComputing = false; // Temporary flag to autofit computed data
PostRanging rangingData[2]; // Data about variation variables and parameters (1 per buffer for stability)
int currentTotalVariations = 0; // Current amount of variations, so we can compare and safely hot-swap the parameter values
bool executedOnLaunch = false; // Temporary flag to execute computations on launch if needed

bool enabledParticles = true; // Particles mode
bool playingParticles = false; // Playing animation
float particleSpeed = 5000.0f; // Steps per second
float particlePhase = 0.0f; // Animation frame cooldown
int particleStep = 0; // Current step of the computations to show
bool continuousComputingEnabled = true; // Continuously compute next batch of steps via double buffering
float dragChangeSpeed = 1.0f;
int bufferNo = 0;

// Plot graph settings
/*
bool markerSettingsWindowEnabled = true;
float markerSize = 1.0f;
float markerOutlineSize = 0.0f;
ImVec4 markerColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
ImPlotMarker markerShape = ImPlotMarker_Circle;
float gridAlpha = 0.15f;
*/

// Temporary variables
int variation = 0;
int stride = 1;
float frameTime; // In seconds
float timeElapsed = 0.0f; // Total time elapsed, in seconds

// Colors
ImVec4 unsavedBackgroundColor = ImVec4(0.427f, 0.427f, 0.137f, 1.0f);
ImVec4 unsavedBackgroundColorHovered = ImVec4(0.427f * 1.3f, 0.427f * 1.3f, 0.137f * 1.3f, 1.0f);
ImVec4 unsavedBackgroundColorActive = ImVec4(0.427f * 1.5f, 0.427f * 1.5f, 0.137f * 1.5f, 1.0f);
ImVec4 disabledColor = ImVec4(0.137f * 0.5f, 0.271f * 0.5f, 0.427f * 0.5f, 1.0f);
ImVec4 disabledTextColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
ImVec4 disabledBackgroundColor = ImVec4(0.137f * 0.35f, 0.271f * 0.35f, 0.427f * 0.35f, 1.0f);
/*ImVec4 xAxisBackgroundColor = ImVec4(0.5f, 0.25f, 0.25f, 1.0f);
ImVec4 yAxisBackgroundColor = ImVec4(0.25f, 0.5f, 0.25f, 1.0f);
ImVec4 zAxisBackgroundColor = ImVec4(0.25f, 0.25f, 0.5f, 1.0f);*/
ImVec4 xAxisColor = ImVec4(0.75f, 0.3f, 0.3f, 1.0f);
ImVec4 yAxisColor = ImVec4(0.33f, 0.67f, 0.4f, 1.0f);
ImVec4 zAxisColor = ImVec4(0.3f, 0.45f, 0.7f, 1.0f);

std::string rangingTypes[] = { "Fixed", "Linear", "Random", "Normal" };

std::future<int> computationFutures[2];

bool rangingWindowEnabled = true;
bool graphBuilderWindowEnabled = true;

// Repetitive stuff
#define LOAD_VARNEW     varNew.load(kernel::VAR_VALUES, kernel::VAR_MAX, kernel::VAR_STEPS, kernel::VAR_RANGING, kernel::VAR_COUNT)
#define UNLOAD_VARNEW   varNew.unload(kernel::VAR_VALUES, kernel::VAR_MAX, kernel::VAR_STEPS, kernel::VAR_RANGING, kernel::VAR_COUNT)
#define LOAD_PARAMNEW   paramNew.load(kernel::PARAM_VALUES, kernel::PARAM_MAX, kernel::PARAM_STEPS, kernel::PARAM_RANGING, kernel::PARAM_COUNT)
#define UNLOAD_PARAMNEW paramNew.unload(kernel::PARAM_VALUES, kernel::PARAM_MAX, kernel::PARAM_STEPS, kernel::PARAM_RANGING, kernel::PARAM_COUNT)
#define PUSH_DISABLED_FRAME {ImGui::PushStyleColor(ImGuiCol_FrameBg, disabledBackgroundColor); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, disabledBackgroundColor); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, disabledBackgroundColor);}
#define PUSH_UNSAVED_FRAME  {ImGui::PushStyleColor(ImGuiCol_FrameBg, unsavedBackgroundColor); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, unsavedBackgroundColorActive); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, unsavedBackgroundColorHovered);}
#define POP_FRAME(n)        {ImGui::PopStyleColor(n);}
#define CLAMP01(x)      if (x < 0.0f) x = 0.0f; if (x > 1.0f) x = 1.0f;

void deleteBothBuffers()
{
    if (computedData[0] != nullptr) { delete[] computedData[0]; computedData[0] = nullptr; }
    if (computedData[1] != nullptr) { delete[] computedData[1]; computedData[1] = nullptr; }

    if (mapBuffers[0] != nullptr) { delete[] mapBuffers[0]; mapBuffers[0] = nullptr; }
    if (mapBuffers[1] != nullptr) { delete[] mapBuffers[1]; mapBuffers[1] = nullptr; }
    //if (compressedHeatmap != nullptr) { delete[] compressedHeatmap; compressedHeatmap = nullptr; }

    computedDataReady[0] = false;
    computedDataReady[1] = false;

    playedBufferIndex = 0;
    bufferToFillIndex = 0;
}

void resetOverrideBuffer(int totalVariations)
{
    if (valuesOverride) delete[] valuesOverride;
    valuesOverride = new numb[kernel::VAR_COUNT * totalVariations];
}

void resetTempBuffers(int totalVariations)
{
    if (dataBuffer) delete[] dataBuffer;
    dataBuffer = new numb[(computedSteps + 1) * kernel::VAR_COUNT];

    if (particleBuffer) delete[] particleBuffer;
    particleBuffer = new numb[totalVariations * kernel::VAR_COUNT];
}

void computing();

int asyncComputation(void** dest, PostRanging* rangingData)
{
    computedDataReady[bufferToFillIndex] = false;

    bool isFirstBatch = computedData[1 - bufferToFillIndex] == nullptr; // Is another buffer null, only true when computing for the first time
    if (isFirstBatch) rangingData->clear();

    //printf("is first batch %i, total variations %i\n", isFirstBatch, rangingData->totalVariations);

    int computationResult = compute(dest, &(mapBuffers[bufferToFillIndex]), isFirstBatch ? nullptr : (numb*)(computedData[1 - bufferToFillIndex]), rangingData);

    computedSteps = kernel::steps;

    if (isFirstBatch)
    {
        autofitAfterComputing = true;
        resetTempBuffers(rangingData->totalVariations);
        resetOverrideBuffer(rangingData->totalVariations);
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
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), LoadIcon(wc.hInstance, MAKEINTRESOURCE(IDI_ICON1)), nullptr, nullptr, nullptr, L"CUDAynamics", LoadIcon(wc.hInstance, MAKEINTRESOURCE(IDI_ICON1)) };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"CUDAynamics", WS_OVERLAPPEDWINDOW, 100, 100, 400, 100, nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

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

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    io.Fonts->AddFontFromFileTTF("UbuntuMono-R.ttf", 24.0f);

    // Main loop
    bool work = true;

    char* plotNameBuffer = new char[64]();
    strcpy_s(plotNameBuffer, 5, "Plot");

    PlotType plotType = Series;
    int selectedPlotVars[3]; selectedPlotVars[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVars[i] = -1;
    set<int> selectedPlotVarsSet;
    int selectedPlotMap = 0;
    LOAD_VARNEW;
    LOAD_PARAMNEW;
    stepsNew = kernel::steps;

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

        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
        {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        timeElapsed += frameTime;
        float breath = (cosf(timeElapsed * 6.0f) + 1.0f) / 2.0f;
        float buttonBreathMult = 1.2f + breath * 0.8f;
        bool noComputedData = computedData[0] == nullptr;

        if (particleStep > computedSteps) particleStep = computedSteps;

        // MAIN WINDOW
        {
            style.WindowMenuButtonPosition = ImGuiDir_Left;
            ImGui::Begin("CUDAynamics", &work);
            ImGui::Text(kernel::name);

            int tempTotalVariations = 1;
            for (int v = 0; v < kernel::VAR_COUNT; v++) if (varNew.RANGING[v]) tempTotalVariations *= (calculateStepCount(varNew.MIN[v], varNew.MAX[v], varNew.STEP[v]));
            for (int p = 0; p < kernel::PARAM_COUNT; p++) if (paramNew.RANGING[p])  tempTotalVariations *= (calculateStepCount(paramNew.MIN[p], paramNew.MAX[p], paramNew.STEP[p]));

            // Parameters & Variables

            ImGuiSliderFlags dragFlag = !playingParticles ? 0 : ImGuiSliderFlags_ReadOnly;

            string namePadded;
            int maxNameLength = 0;

            for (int i = 0; i < kernel::PARAM_COUNT; i++) if (strlen(kernel::PARAM_NAMES[i]) > maxNameLength) maxNameLength = (int)strlen(kernel::PARAM_NAMES[i]);
            for (int i = 0; i < kernel::VAR_COUNT; i++) if (strlen(kernel::VAR_NAMES[i]) > maxNameLength) maxNameLength = (int)strlen(kernel::VAR_NAMES[i]);          

            bool anyChanged = false;
            bool thisChanged;
            bool popStyle;

            ImGui::SeparatorText("Variables");

            for (int i = 0; i < kernel::VAR_COUNT; i++)
            {
                thisChanged = false;
                if (varNew.MIN[i] != kernel::VAR_VALUES[i]) { anyChanged = true; thisChanged = true; }
                if (varNew.MAX[i] != kernel::VAR_MAX[i]) { anyChanged = true; thisChanged = true; }
                if (varNew.STEP[i] != kernel::VAR_STEPS[i]) { anyChanged = true; thisChanged = true; }
                if (varNew.RANGING[i] != kernel::VAR_RANGING[i]) { anyChanged = true; thisChanged = true; }
                if (thisChanged) varNew.recountSteps(i);

                namePadded = kernel::VAR_NAMES[i];
                for (int j = (int)strlen(kernel::VAR_NAMES[i]); j < maxNameLength; j++)
                    namePadded += ' ';

                ImGui::Text(namePadded.c_str());

                if (playingParticles)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
                    PUSH_DISABLED_FRAME;
                }

                // Ranging
                ImGui::SameLine();
                popStyle = false;
                if (varNew.RANGING[i] != kernel::VAR_RANGING[i])
                {
                    PUSH_UNSAVED_FRAME;
                    popStyle = true;
                }
                ImGui::PushItemWidth(120.0f);
                if (ImGui::BeginCombo(("##RANGING_" + std::string(kernel::VAR_NAMES[i])).c_str(), (rangingTypes[varNew.RANGING[i]]).c_str()))
                {
                    for (int r = 0; r < 4; r++)
                    {
                        bool isSelected = varNew.RANGING[i] == r;
                        ImGuiSelectableFlags selectableFlags = 0;
                        if (ImGui::Selectable(rangingTypes[r].c_str(), isSelected, selectableFlags)) varNew.RANGING[i] = (RangingType)r;
                    }

                    ImGui::EndCombo();
                }
                ImGui::PopItemWidth();
                if (popStyle) POP_FRAME(3);

                // Min
                ImGui::SameLine();
                ImGui::PushItemWidth(150.0f);
                popStyle = false;
                if (varNew.MIN[i] != kernel::VAR_VALUES[i])
                {
                    PUSH_UNSAVED_FRAME;
                    popStyle = true;
                }
                float varNewMin = (float)varNew.MIN[i];
                ImGui::DragFloat(("##" + std::string(kernel::VAR_NAMES[i])).c_str(), &varNewMin, dragChangeSpeed, 0.0f, 0.0f, "%f", dragFlag);
                varNew.MIN[i] = (double)varNewMin;
                if (popStyle) POP_FRAME(3);
                ImGui::PopItemWidth();

                // If ranging
                if (varNew.RANGING[i])
                {
                    // Step
                    ImGui::SameLine();
                    ImGui::PushItemWidth(150.0f);
                    popStyle = false;
                    if (varNew.STEP[i] != kernel::VAR_STEPS[i])
                    {
                        PUSH_UNSAVED_FRAME;
                        popStyle = true;
                    }
                    float varNewStep = (float)varNew.STEP[i];
                    ImGui::DragFloat(("##STEP_" + std::string(kernel::VAR_NAMES[i])).c_str(), &varNewStep, dragChangeSpeed, 0.0f, 0.0f, "%f", dragFlag);
                    varNew.STEP[i] = (double)varNewStep;
                    if (popStyle) POP_FRAME(3);

                    // Max
                    ImGui::SameLine();
                    popStyle = false;
                    if (varNew.MAX[i] != kernel::VAR_MAX[i])
                    {
                        PUSH_UNSAVED_FRAME;
                        popStyle = true;
                    }
                    float varNewMax = (float)varNew.MAX[i];
                    ImGui::DragFloat(("##MAX_" + std::string(kernel::VAR_NAMES[i])).c_str(), &varNewMax, dragChangeSpeed, 0.0f, 0.0f, "%f", dragFlag);
                    varNew.MAX[i] = (double)varNewMax;
                    if (popStyle) POP_FRAME(3);
                    ImGui::PopItemWidth();

                    // Step count
                    ImGui::SameLine();
                    ImGui::Text((std::to_string(calculateStepCount(varNew.MIN[i], varNew.MAX[i], varNew.STEP[i])) + " steps").c_str());
                }

                if (playingParticles)
                {
                    ImGui::PopStyleColor();
                    POP_FRAME(3);
                }
            }

            ImGui::SeparatorText("Parameters");

            bool applicationProhibited = false;

            for (int i = 0; i < kernel::PARAM_COUNT; i++)
            {
                bool isRanging = paramNew.RANGING[i];
                bool changeAllowed = !paramNew.RANGING[i] || !playingParticles || !autoLoadNewParams;

                thisChanged = false;
                if (paramNew.MIN[i] != kernel::PARAM_VALUES[i]) { anyChanged = true; thisChanged = true; }
                if (paramNew.MAX[i] != kernel::PARAM_MAX[i]) { anyChanged = true; thisChanged = true; }
                if (paramNew.STEP[i] != kernel::PARAM_STEPS[i]) { anyChanged = true; thisChanged = true; }
                if (paramNew.RANGING[i] != kernel::PARAM_RANGING[i]) { anyChanged = true; thisChanged = true; }
                if (thisChanged) paramNew.recountSteps(i);

                namePadded = kernel::PARAM_NAMES[i];
                for (int j = (int)strlen(kernel::PARAM_NAMES[i]); j < maxNameLength; j++)
                    namePadded += ' ';

                ImGui::Text(namePadded.c_str());

                if (!changeAllowed)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor); // disabledText push
                    PUSH_DISABLED_FRAME;
                }

                // Ranging
                ImGui::SameLine();
                if (playingParticles) PUSH_DISABLED_FRAME;
                popStyle = false;
                if (paramNew.RANGING[i] != kernel::PARAM_RANGING[i])
                {
                    PUSH_UNSAVED_FRAME;
                    popStyle = true;
                }
                ImGui::PushItemWidth(120.0f);
                if (ImGui::BeginCombo(("##RANGING_" + std::string(kernel::PARAM_NAMES[i])).c_str(), (rangingTypes[paramNew.RANGING[i]]).c_str()))
                {
                    for (int r = 0; r < 4; r++)
                    {
                        bool isSelected = paramNew.RANGING[i] == r;
                        ImGuiSelectableFlags selectableFlags = 0;
                        if (ImGui::Selectable(rangingTypes[r].c_str(), isSelected, selectableFlags)) paramNew.RANGING[i] = (RangingType)r;
                    }

                    ImGui::EndCombo();
                }
                ImGui::PopItemWidth();
                if (popStyle) POP_FRAME(3);
                if (playingParticles) POP_FRAME(3);

                // Min
                ImGui::SameLine();
                ImGui::PushItemWidth(150.0f);
                popStyle = false;
                if (paramNew.MIN[i] != kernel::PARAM_VALUES[i])
                {
                    PUSH_UNSAVED_FRAME;
                    popStyle = true;
                }
                float paramNewMin = (float)paramNew.MIN[i];
                ImGui::DragFloat(("##" + std::string(kernel::PARAM_NAMES[i])).c_str(), &paramNewMin, dragChangeSpeed, 0.0f, 0.0f, "%f", changeAllowed ? 0 : ImGuiSliderFlags_ReadOnly);
                paramNew.MIN[i] = (double)paramNewMin;
                if (popStyle) POP_FRAME(3);
                ImGui::PopItemWidth();

                // If ranging
                if (paramNew.RANGING[i])
                {
                    // Step
                    ImGui::SameLine();
                    ImGui::PushItemWidth(150.0f);
                    popStyle = false;
                    if (paramNew.STEP[i] != kernel::PARAM_STEPS[i])
                    {
                        PUSH_UNSAVED_FRAME;
                        popStyle = true;
                    }
                    float paramNewStep = (float)paramNew.STEP[i];
                    ImGui::DragFloat(("##STEP_" + std::string(kernel::PARAM_NAMES[i])).c_str(), &paramNewStep, dragChangeSpeed, 0.0f, 0.0f, "%f", changeAllowed ? 0 : ImGuiSliderFlags_ReadOnly);
                    paramNew.STEP[i] = (double)paramNewStep;
                    if (popStyle) POP_FRAME(3);

                    // Max
                    ImGui::SameLine();
                    popStyle = false;
                    if (paramNew.MAX[i] != kernel::PARAM_MAX[i])
                    {
                        PUSH_UNSAVED_FRAME;
                        popStyle = true;
                    }
                    float paramNewMax = (float)paramNew.MAX[i];
                    ImGui::DragFloat(("##MAX_" + std::string(kernel::PARAM_NAMES[i])).c_str(), &paramNewMax, dragChangeSpeed, 0.0f, 0.0f, "%f", changeAllowed ? 0 : ImGuiSliderFlags_ReadOnly);
                    paramNew.MAX[i] = (double)paramNewMax;
                    if (popStyle) POP_FRAME(3);
                    ImGui::PopItemWidth();
                }

                if (!changeAllowed) POP_FRAME(4); // disabledText popped as well

                // Step count
                if (paramNew.RANGING[i])
                {
                    int stepCount = calculateStepCount(kernel::PARAM_VALUES[i], kernel::PARAM_MAX[i], kernel::PARAM_STEPS[i]);
                    if (stepCount > 0)
                    {
                        ImGui::SameLine();
                        ImGui::Text((std::to_string(stepCount) + " steps").c_str());

                        if (thisChanged && stepCount != paramNew.stepsOf(i))
                        {
                            ImGui::SameLine();
                            ImGui::Text(("(new - " + std::to_string(paramNew.stepsOf(i)) + " steps)").c_str());
                            applicationProhibited = true;
                        }
                    }
                }
            }

            if (enabledParticles)
            {
                bool tempAutoLoadNewParams = autoLoadNewParams;
                if (ImGui::Checkbox("Apply parameter changes automatically", &(tempAutoLoadNewParams)))
                {
                    autoLoadNewParams = !autoLoadNewParams;
                    if (autoLoadNewParams) LOAD_PARAMNEW;
                    else UNLOAD_PARAMNEW;
                }

                ImGui::PushItemWidth(200.0f);
                ImGui::InputFloat("Drag speed", &(dragChangeSpeed));
            }

            if (autoLoadNewParams)
            {
                UNLOAD_PARAMNEW;
            }
            else if (playingParticles && !autoLoadNewParams && anyChanged)
            {
                if (applicationProhibited)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
                    ImGui::PushStyleColor(ImGuiCol_Button, disabledBackgroundColor);
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, disabledBackgroundColor);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, disabledBackgroundColor);
                }
                if (ImGui::Button("Apply") && !applicationProhibited)
                {
                    UNLOAD_PARAMNEW;
                }
                if (applicationProhibited)
                {
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();
                }
            }

            // Simulation

            ImGui::SeparatorText("Simulation");

            unsigned long long tempTotalVariationsLL = tempTotalVariations;
            unsigned long long varCountLL = kernel::VAR_COUNT;
            unsigned long long stepsNewLL = stepsNew + 1;
            unsigned long long singleBufferNumberCount = ((tempTotalVariationsLL * varCountLL) * stepsNewLL);
            unsigned long long singleBufferNumbSize = singleBufferNumberCount * sizeof(numb);
            ImGui::Text(("Single buffer size: " + memoryString(singleBufferNumbSize) + " (" + to_string(singleBufferNumbSize) + " bytes)").c_str());

            ImGui::PushItemWidth(200.0f);
            if (playingParticles)
            {
                ImGui::PushStyleColor(ImGuiCol_FrameBg, disabledBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, disabledBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, disabledBackgroundColor);
            }
            popStyle = false;
            if (stepsNew != kernel::steps)
            {
                anyChanged = true;
                PUSH_UNSAVED_FRAME;
                popStyle = true;
            }
            ImGui::InputInt("Steps", &(stepsNew), 1, 1000, playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
            if (popStyle) POP_FRAME(3);
            if (playingParticles)
            {
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
            }
            float tempStepSize = (float)kernel::stepSize;
            ImGui::InputFloat("Step size", &tempStepSize, 0.0f, 0.0f, "%f");
            kernel::stepSize = (double)tempStepSize;
            ImGui::PopItemWidth();
            
            bool tempEnabledParticles = enabledParticles;
            if (ImGui::Checkbox("Particles mode", &(tempEnabledParticles)))
            {
                enabledParticles = !enabledParticles;
            }

            if (tempEnabledParticles)
            {
                ImGui::PushItemWidth(200.0f);
                ImGui::DragFloat("Animation speed, steps/s", &(particleSpeed), 1.0f);
                if (particleSpeed < 0.0f) particleSpeed = 0.0f;
                ImGui::PopItemWidth();

                if (rangingData[playedBufferIndex].timeElapsed > 0.0f)
                {
                    float buffersPerSecond = 1000.0f / rangingData[playedBufferIndex].timeElapsed;
                    int stepsPerSecond = (int)(computedSteps * buffersPerSecond);

                    ImGui::SameLine();
                    ImGui::Text(("(max " + to_string(stepsPerSecond) + " before stalling)").c_str());
                }

                ImGui::DragInt("##Animation step", &(particleStep), 1.0f, 0, kernel::steps);
                ImGui::SameLine();
                ImGui::Text(("Animation step" + (continuousComputingEnabled ? " (total step " + to_string(bufferNo * kernel::steps + particleStep) + ")" : "")).c_str());

                if (ImGui::Button("Reset to step 0"))
                {
                    particleStep = 0;
                }

                if (anyChanged)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
                    PUSH_DISABLED_FRAME;
                }
                bool tempPlayingParticles = playingParticles;
                if (ImGui::Checkbox("Play", &(tempPlayingParticles)) && !anyChanged)
                {
                    if (computedDataReady[0] || playingParticles)
                    {
                        playingParticles = !playingParticles;
                        LOAD_PARAMNEW;
                    }

                    if (!playingParticles)
                    {
                        UNLOAD_PARAMNEW;
                    }
                }
                if (anyChanged) POP_FRAME(4);

                bool tempContinuous = continuousComputingEnabled;
                if (ImGui::Checkbox("Continuous computing", &(tempContinuous)))
                {
                    // Flags of having buffers computed, to not interrupt computations in progress when switching
                    bool noncont = !continuousComputingEnabled && computedDataReady[0];
                    bool cont = continuousComputingEnabled && computedDataReady[0] && computedDataReady[1];

                    if (noComputedData || noncont || cont)
                    {
                        continuousComputingEnabled = !continuousComputingEnabled;

                        bufferNo = 0;
                        deleteBothBuffers();
                        playingParticles = false;
                        particleStep = 0;
                    }
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
                            bufferNo++;
                            //printf("Switch occured and starting playing %i\n", playedBufferIndex);
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
            bool playBreath = noComputedData || (anyChanged && (!playingParticles || !enabledParticles));
            if (playBreath)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.137f * buttonBreathMult, 0.271f * buttonBreathMult, 0.427f * buttonBreathMult, 1.0f));
            if (ImGui::Button("= COMPUTE =") || (kernel::executeOnLaunch && !executedOnLaunch))
            {
                executedOnLaunch = true;
                bufferToFillIndex = 0;
                playedBufferIndex = 0;
                bufferNo = 0;
                deleteBothBuffers();

                UNLOAD_VARNEW;
                UNLOAD_PARAMNEW;
                kernel::steps = stepsNew;

                computing();
            }
            if (playBreath) ImGui::PopStyleColor();

            if (anyChanged)
            {
                if (ImGui::Button("Reset changed values"))
                {
                    LOAD_VARNEW;
                    LOAD_PARAMNEW;
                }
            }

            ImGui::End();

            // RANGING

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
                        numb currentValue = calculateValue(rangingData[playedBufferIndex].min[r], rangingData[playedBufferIndex].step[r], rangingData[playedBufferIndex].currentStep[r]);
                        rangingData[playedBufferIndex].currentValue[r] = currentValue;

                        ImGui::PushItemWidth(150.0f);
                        ImGui::DragInt(("##" + rangingData[playedBufferIndex].names[r] + "_ranging").c_str(), &(rangingData[playedBufferIndex].currentStep[r]), 1.0f, 0, rangingData[playedBufferIndex].stepCount[r] - 1);
                        ImGui::SameLine();
                        ImGui::Text((rangingData[playedBufferIndex].names[r] + " = " + std::to_string(currentValue)).c_str());
                        ImGui::PopItemWidth();
                    }

                    ImGui::Text(("Current variations: " + std::to_string(currentTotalVariations)).c_str());

                    // Apply ranging configuration as fixed values
                    if (ImGui::Button("Apply fixed values"))
                    {
                        for (int r = 0; r < rangingData[playedBufferIndex].rangingCount; r++)
                        {
                            bool isParam = false;
                            int entityIndex = -1;
                            rangingData[playedBufferIndex].getIndexOfVarOrParam(&isParam, &entityIndex, kernel::VAR_COUNT, kernel::PARAM_COUNT, &(kernel::VAR_NAMES[0]), &(kernel::PARAM_NAMES[0]), r);

                            if (entityIndex == -1) continue;

                            if (!isParam)
                            {
                                varNew.RANGING[entityIndex] = RangingType::None;
                                varNew.MIN[entityIndex] = rangingData[playedBufferIndex].currentValue[r];
                            }
                            else
                            {
                                paramNew.RANGING[entityIndex] = RangingType::None;
                                paramNew.MIN[entityIndex] = rangingData[playedBufferIndex].currentValue[r];
                            }
                        }
                    }

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

        bool toAutofit = autofitAfterComputing;
        autofitAfterComputing = false;

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

            bool autofitHeatmap = false;

            // Plot variables
            if (ImGui::BeginCombo(("##" + windowName + "_plotSettings").c_str(), "Plot settings"))
            {
                bool tempWhiteBg = window->whiteBg; ImGui::SameLine(); if (ImGui::Checkbox(("##" + windowName + "whiteBG").c_str(), &tempWhiteBg)) window->whiteBg = !window->whiteBg;
                ImGui::SameLine(); ImGui::Text("White background");

                bool tempGrayscale = window->grayscaleHeatmap; ImGui::SameLine(); if (ImGui::Checkbox(("##" + windowName + "grayscale").c_str(), &tempGrayscale)) window->grayscaleHeatmap = !window->grayscaleHeatmap;
                ImGui::SameLine(); ImGui::Text("Grayscale");

                if (window->type == Phase || window->type == Series)
                {
                    ImGui::DragFloat(("##" + windowName + "_markerSize").c_str(), &(window->markerSize), 0.1f);                             ImGui::SameLine(); ImGui::Text("Marker size");
                    ImGui::DragFloat(("##" + windowName + "_markerOutlineSize").c_str(), &(window->markerOutlineSize), 0.1f);               ImGui::SameLine(); ImGui::Text("Marker outline size");
                    if (window->markerSize < 0.0f) window->markerSize = 0.0f;
                    ImGui::ColorEdit4(("##" + windowName + "_markerColor").c_str(), (float*)(&(window->markerColor)));                      ImGui::SameLine(); ImGui::Text("Marker color");

                    /*std::string shapeNames[]{ "Circle", "Square", "Diamond", "Up", "Down", "Left", "Right", "Cross", "Plus", "Asterisk" };
                    if (ImGui::BeginCombo(("##" + windowName + "_markerShape").c_str(), shapeNames[window->markerShape].c_str()))
                    {
                        for (ImPlotMarker i = 0; i < ImPlotMarker_COUNT; i++)
                        {
                            bool isSelected = window->markerShape == i;
                            if (ImGui::Selectable(shapeNames[i].c_str(), isSelected)) window->markerShape = i;
                        }
                        ImGui::EndCombo();
                    }                                                                                                               ImGui::SameLine(); ImGui::Text("Marker shape");*/

                    bool tempShowAxis = window->showAxis; if (ImGui::Checkbox(("##" + windowName + "showAxis").c_str(), &tempShowAxis)) window->showAxis = !window->showAxis;
                    ImGui::SameLine(); ImGui::Text("Show axis"); ImGui::SameLine();
                    bool tempShowAxisNames = window->showAxisNames; if (ImGui::Checkbox(("##" + windowName + "showAxisNames").c_str(), &tempShowAxisNames)) window->showAxisNames = !window->showAxisNames;
                    ImGui::SameLine(); ImGui::Text("Show axis names");

                    ImGui::DragFloat(("##" + windowName + "_rulerAlpha").c_str(), &(window->rulerAlpha), 0.01f);
                    bool tempShowRuler = window->showRuler; ImGui::SameLine(); if (ImGui::Checkbox(("##" + windowName + "showRuler").c_str(), &tempShowRuler)) window->showRuler = !window->showRuler;
                    ImGui::SameLine(); ImGui::Text("Ruler alpha");
                    CLAMP01(window->rulerAlpha);
                    ImGui::DragFloat(("##" + windowName + "_gridAlpha").c_str(), &(window->gridAlpha), 0.01f);
                    bool tempShowGrid = window->showGrid; ImGui::SameLine(); if (ImGui::Checkbox(("##" + windowName + "showGrid").c_str(), &tempShowGrid)) window->showGrid = !window->showGrid;
                    ImGui::SameLine(); ImGui::Text("Grid alpha");
                    CLAMP01(window->gridAlpha);
                }

                if (window->type == Heatmap)
                {
                    ImGui::DragInt(("##" + windowName + "_stride").c_str(), (int*)(&(window->stride)), 1.0f); ImGui::SameLine(); ImGui::Text("Stride");
                    if (window->stride < 1) window->stride = 1;

                    bool tempShowHeatmapValues = window->showHeatmapValues; if (ImGui::Checkbox(("##" + windowName + "showHeatmapValues").c_str(), &tempShowHeatmapValues)) window->showHeatmapValues = !window->showHeatmapValues;
                    ImGui::SameLine(); ImGui::Text("Show values");

                    bool tempShowActualDiapasons = window->showActualDiapasons; if (ImGui::Checkbox(("##" + windowName + "showActualDiapasons").c_str(), &tempShowActualDiapasons))
                    {
                        window->showActualDiapasons = !window->showActualDiapasons;
                        autofitHeatmap = true;
                    }
                    ImGui::SameLine(); ImGui::Text("Value diapasons");

                    bool tempHeatmapSelectionMode = window->isHeatmapSelectionModeOn; if (ImGui::Checkbox(("##" + windowName + "heatmapSelectionMode").c_str(), &tempHeatmapSelectionMode)) window->isHeatmapSelectionModeOn = !window->isHeatmapSelectionModeOn;
                    ImGui::SameLine(); ImGui::Text("Selection mode");
                }

                ImGui::EndCombo();
            }

            // Common variables
            ImPlotAxisFlags axisFlags = (toAutofit ? ImPlotAxisFlags_AutoFit : 0);
            ImPlotPlot* plot;
            bool is3d;
            int mapIndex;
            ImPlotColormap heatmapColorMap = !window->grayscaleHeatmap ? ImPlotColormap_Jet : ImPlotColormap_Greys;
            ImVec4 rotationEuler;

            switch (window->type)
            {
            case Series:

                //printf("Begin series\n");
                if (window->whiteBg) ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle, axisFlags, axisFlags))
                {
                    plot = ImPlot::GetPlot(plotName.c_str());

                    plot->is3d = false;

                    if (computedDataReady[playedBufferIndex])
                    {
                        int variationSize = kernel::VAR_COUNT * (computedSteps + 1);

                        void* computedVariation = (numb*)(computedData[playedBufferIndex]) + (variationSize * variation);
                        memcpy(dataBuffer, computedVariation, variationSize * sizeof(numb));

                        //void PlotLine(const char* label_id, const T* values, int count, double xscale, double x0, ImPlotLineFlags flags, int offset, int stride)

                        for (int v = 0; v < window->variableCount; v++)
                        {
                            //ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                            ImPlot::PlotLine((std::string(kernel::VAR_NAMES[window->variables[v]]) + "##" + plotName + std::to_string(v)).c_str(),
                                &(((numb*)dataBuffer)[window->variables[v]]), computedSteps + 1, 1.0f, 0.0f, ImPlotLineFlags_None, 0, sizeof(numb) * kernel::VAR_COUNT);
                        }
                    }

                    //printf("End series\n");
                    ImPlot::EndPlot();
                }
                if (window->whiteBg) ImPlot::PopStyleColor();

                break;

            case Phase:
                // PHASE DIAGRAM
                is3d = window->variableCount == 3;
                rotationEuler = ToEulerAngles(window->quatRot);

                if (is3d)
                {
                    ImVec4 rotationEulerEditable(rotationEuler);                 
                    rotationEulerEditable.x /= DEG2RAD;
                    rotationEulerEditable.y /= DEG2RAD;
                    rotationEulerEditable.z /= DEG2RAD;
                    ImVec4 rotationEulerBeforeEdit(rotationEulerEditable);

                    rotationEulerEditable.x += window->autorotate.x * frameTime;
                    rotationEulerEditable.y += window->autorotate.y * frameTime;
                    rotationEulerEditable.z += window->autorotate.z * frameTime;
                    
                    ImGui::DragFloat3("Rotation", (float*)&rotationEulerEditable, 1.0f);
                    ImGui::DragFloat3("Automatic rotation", (float*)&window->autorotate, 0.1f);
                    ImGui::DragFloat3("Offset", (float*)&window->offset, 0.01f);
                    ImGui::DragFloat3("Scale", (float*)&window->scale, 0.01f);

                    if (window->scale.x < 0.0f) window->scale.x = 0.0f;
                    if (window->scale.y < 0.0f) window->scale.y = 0.0f;
                    if (window->scale.z < 0.0f) window->scale.z = 0.0f;

                    if (rotationEulerBeforeEdit != rotationEulerEditable)
                    {
                        // Rotate quaternion by euler drag

                        ImVec4 deltaEuler = rotationEulerEditable - rotationEulerBeforeEdit;

                        quaternion::Quaternion<float> quatEditable(1.0f, 0.0f, 0.0f, 0.0f);
                        quaternion::Quaternion<float> quatRot(window->quatRot.w, window->quatRot.x, window->quatRot.y, window->quatRot.z);
                        quaternion::Quaternion<float> quatZ(cosf(deltaEuler.z * 0.5f * DEG2RAD), 0.0f, 0.0f, sinf(deltaEuler.z * 0.5f * DEG2RAD));
                        quaternion::Quaternion<float> quatY(cosf(deltaEuler.y * 0.5f * DEG2RAD), 0.0f, sinf(deltaEuler.y * 0.5f * DEG2RAD), 0.0f);
                        quaternion::Quaternion<float> quatX(cosf(deltaEuler.x * 0.5f * DEG2RAD), sinf(deltaEuler.x * 0.5f * DEG2RAD), 0.0f, 0.0f);

                        if (deltaEuler.x != 0.0f) quatEditable = quatX * quatEditable;
                        if (deltaEuler.y != 0.0f) quatEditable = quatY * quatEditable;
                        if (deltaEuler.z != 0.0f) quatEditable = quatZ * quatEditable;

                        quatEditable = quatRot * quatEditable;

                        window->quatRot.w = quatEditable.a();
                        window->quatRot.x = quatEditable.b();
                        window->quatRot.y = quatEditable.c();
                        window->quatRot.z = quatEditable.d();
                    }

                    axisFlags |= ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels;
                }

                //printf("Begin phase\n");
                if (window->whiteBg) { ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); }
                if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, axisFlags, axisFlags))
                {
                    plot = ImPlot::GetPlot(plotName.c_str());

                    float plotRangeSize = (float)plot->Axes[ImAxis_X1].Range.Max - (float)plot->Axes[ImAxis_X1].Range.Min;

                    float deltax = -window->deltarotation.x;
                    float deltay = -window->deltarotation.y;

                    window->deltarotation.x = 0;
                    window->deltarotation.y = 0;

                    plot->is3d = is3d;
                    plot->deltax = &(window->deltarotation.x);
                    plot->deltay = &(window->deltarotation.y);

                    if (deltax != 0.0f || deltay != 0.0f)
                    {
                        quaternion::Quaternion<float> quat(window->quatRot.w, window->quatRot.x, window->quatRot.y, window->quatRot.z);
                        quaternion::Quaternion<float> quatY(cosf(deltax * 0.5f * DEG2RAD), 0.0f, sinf(deltax * 0.5f * DEG2RAD), 0.0f);
                        quaternion::Quaternion<float> quatX(cosf(deltay * 0.5f * DEG2RAD), sinf(deltay * 0.5f * DEG2RAD), 0.0f, 0.0f);
                        quat = quatY * quatX * quat;
                        quat = quaternion::normalize(quat);
                        window->quatRot.w = quat.a();
                        window->quatRot.x = quat.b();
                        window->quatRot.y = quat.c();
                        window->quatRot.z = quat.d();
                    }
                    //printf("%f %f %f %f\n", window->quatRot.x, window->quatRot.y, window->quatRot.z, window->quatRot.w);
                    rotationEuler = ToEulerAngles(window->quatRot);

                    if (computedDataReady[playedBufferIndex])
                    {
                        int variationSize = kernel::VAR_COUNT * (computedSteps + 1);

                        populateAxisBuffer((numb*)axisBuffer, plotRangeSize / 10, plotRangeSize / 10, plotRangeSize / 10);
                        if (is3d)
                        {
                            rotateOffsetBuffer((numb*)axisBuffer, 6, 3, 0, 1, 2, rotationEuler, ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 0));
                        }

                        int xIndex = is3d ? 0 : window->variables[0];
                        int yIndex = is3d ? 1 : window->variables[1];

                        if (!enabledParticles) // Trajectory - one variation, all steps
                        {
                            void* computedVariation = (numb*)(computedData[playedBufferIndex]) + (variationSize * variation);
                            memcpy(dataBuffer, computedVariation, variationSize * sizeof(numb));

                            if (is3d)
                                rotateOffsetBuffer((numb*)dataBuffer, computedSteps + 1, kernel::VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                    rotationEuler, window->offset, window->scale);

                            getMinMax2D((numb*)dataBuffer, computedSteps + 1, &(plot->dataMin), &(plot->dataMax));
                            //printf("%f:%f %f:%f\n", plot->dataMin.x, plot->dataMin.y, plot->dataMax.x, plot->dataMax.y);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                            ImPlot::PlotLine(plotName.c_str(), &(((numb*)dataBuffer)[xIndex]), &(((numb*)dataBuffer)[yIndex]), computedSteps + 1, 0, 0, sizeof(numb) * kernel::VAR_COUNT);
                        }
                        else // Particles - all variations, one certain step
                        {
                            if (particleStep > kernel::steps) particleStep = kernel::steps;

                            for (int v = 0; v < rangingData[playedBufferIndex].totalVariations; v++)
                            {
                                for (int var = 0; var < kernel::VAR_COUNT; var++)
                                    ((numb*)particleBuffer)[v * kernel::VAR_COUNT + var] = ((numb*)(computedData[playedBufferIndex]))[(variationSize * v) + (kernel::VAR_COUNT * particleStep) + var];
                            }

                            if (is3d)
                                rotateOffsetBuffer((numb*)particleBuffer, rangingData[playedBufferIndex].totalVariations, kernel::VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                    rotationEuler, window->offset, window->scale);

                            ImPlot::SetNextLineStyle(window->markerColor);
                            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineSize);
                            //ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
                            ImPlot::SetNextMarkerStyle(window->markerShape, window->markerSize);
                            ImPlot::PlotScatter(plotName.c_str(), &(((numb*)particleBuffer)[xIndex]), &(((numb*)particleBuffer)[yIndex]), rangingData[playedBufferIndex].totalVariations, 0, 0, sizeof(numb) * kernel::VAR_COUNT);
                        }

                        // Axis
                        if (window->showAxis)
                        {
                            ImPlot::SetNextLineStyle(xAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(((numb*)axisBuffer)[0]), &(((numb*)axisBuffer)[1]), 2, 0, 0, sizeof(numb) * 3);
                            ImPlot::SetNextLineStyle(yAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(((numb*)axisBuffer)[6]), &(((numb*)axisBuffer)[7]), 2, 0, 0, sizeof(numb) * 3);

                            if (is3d)
                            {
                                ImPlot::SetNextLineStyle(zAxisColor);
                                ImPlot::PlotLine(plotName.c_str(), &(((numb*)axisBuffer)[12]), &(((numb*)axisBuffer)[13]), 2, 0, 0, sizeof(numb) * 3);
                            }
                        }

                        // Axis names
                        if (window->showAxisNames)
                        {
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, xAxisColor);
                            ImPlot::PlotText(kernel::VAR_NAMES[window->variables[0]], ((numb*)axisBuffer)[0], ((numb*)axisBuffer)[1], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, yAxisColor);
                            ImPlot::PlotText(kernel::VAR_NAMES[window->variables[1]], ((numb*)axisBuffer)[6], ((numb*)axisBuffer)[7], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();

                            if (is3d)
                            {
                                ImPlot::PushStyleColor(ImPlotCol_InlayText, zAxisColor);
                                ImPlot::PlotText(kernel::VAR_NAMES[window->variables[2]], ((numb*)axisBuffer)[12], ((numb*)axisBuffer)[13], ImVec2(0.0f, 0.0f));
                                ImPlot::PopStyleColor();
                            }
                        }

                        // Ruler
                        if (is3d && window->showRuler)
                        {
                            ImVec4 scale(plotRangeSize / window->scale.x, plotRangeSize / window->scale.y, plotRangeSize / window->scale.z, 0);
                            ImVec4 scaleLog(floorf(log10f(scale.x)), floorf(log10f(scale.y)), floorf(log10f(scale.z)), 0);
                            ImVec4 scale0(powf(10, scaleLog.x - 1), powf(10, scaleLog.y - 1), powf(10, scaleLog.z - 1), 0);
                            ImVec4 scale1(powf(10, scaleLog.x), powf(10, scaleLog.y), powf(10, scaleLog.z), 0);
                            ImVec4 scaleInterp(log10f(scale.x) - scaleLog.x, log10f(scale.y) - scaleLog.y, log10f(scale.z) - scaleLog.z, 0);

                            ImVec4 alpha0((1.0f - scaleInterp.x) * window->rulerAlpha, (1.0f - scaleInterp.y) * window->rulerAlpha, (1.0f - scaleInterp.z) * window->rulerAlpha, 0);
                            ImVec4 alpha1(scaleInterp.x * window->rulerAlpha, scaleInterp.y * window->rulerAlpha, scaleInterp.z * window->rulerAlpha, 0);

#define DRAW_RULER_PART(colorR, colorG, colorB, alpha, scale, scaleStr, dim) ImPlot::SetNextLineStyle(ImVec4(colorR, colorG, colorB, alpha)); \
                            populateRulerBuffer((numb*)rulerBuffer, scale, dim); \
                            rotateOffsetBuffer((numb*)rulerBuffer, 51, 3, 0, 1, 2, rotationEuler, \
                                ImVec4(0, 0, 0, 0), ImVec4(scale, scale, scale, 0)); \
                            ImPlot::PlotLine(plotName.c_str(), &(((numb*)rulerBuffer)[0]), &(((numb*)rulerBuffer)[1]), 51, 0, 0, sizeof(numb) * 3); \
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, ImVec4(colorR, colorG, colorB, alpha)); \
                            ImPlot::PlotText(scaleString(scaleStr).c_str(), ((numb*)rulerBuffer)[150 + 0], ((numb*)rulerBuffer)[150 + 1], ImVec2(0.0f, 0.0f)); \
                            ImPlot::PopStyleColor();

                            DRAW_RULER_PART(xAxisColor.x, xAxisColor.y, xAxisColor.z, alpha0.x, scale0.x * window->scale.x, scale0.x, 0);
                            DRAW_RULER_PART(xAxisColor.x, xAxisColor.y, xAxisColor.z, alpha1.x, scale1.x* window->scale.x, scale1.x, 0);

                            DRAW_RULER_PART(yAxisColor.x, yAxisColor.y, yAxisColor.z, alpha0.y, scale0.y* window->scale.y, scale0.y, 1);
                            DRAW_RULER_PART(yAxisColor.x, yAxisColor.y, yAxisColor.z, alpha1.y, scale1.y* window->scale.y, scale1.y, 1);

                            DRAW_RULER_PART(zAxisColor.x, zAxisColor.y, zAxisColor.z, alpha0.z, scale0.z * window->scale.z, scale0.z, 2);
                            DRAW_RULER_PART(zAxisColor.x, zAxisColor.y, zAxisColor.z, alpha1.z, scale1.z * window->scale.z, scale1.z, 2);
                        }

                        // Grid
#if 0
                        if (is3d && window->showGrid)
                        {
                            ImVec4 scale(plotRangeSize / window->scale.x, plotRangeSize / window->scale.y, plotRangeSize / window->scale.z, 0);
                            ImVec4 scaleLog(floorf(log10f(scale.x)), floorf(log10f(scale.y)), floorf(log10f(scale.z)), 0);
                            ImVec4 scale0(powf(10, scaleLog.x - 1), powf(10, scaleLog.y - 1), powf(10, scaleLog.z - 1), 0);
                            ImVec4 scale1(powf(10, scaleLog.x), powf(10, scaleLog.y), powf(10, scaleLog.z), 0);
                            ImVec4 scaleInterp(log10f(scale.x) - scaleLog.x, log10f(scale.y) - scaleLog.y, log10f(scale.z) - scaleLog.z, 0);

                            ImVec4 alpha0((1.0f - scaleInterp.x) * window->gridAlpha, (1.0f - scaleInterp.y) * window->gridAlpha, (1.0f - scaleInterp.z) * window->gridAlpha, 0);
                            ImVec4 alpha1(scaleInterp.x * window->gridAlpha, scaleInterp.y * window->gridAlpha, scaleInterp.z * window->gridAlpha, 0);

                            // x
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha0.x));
                            populateGridBuffer((float*)gridBuffer);
                            rotateOffsetBuffer2((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale0.x * window->scale.x, scale0.y * window->scale.y, scale0.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                            /*ImPlot::PushStyleColor(ImPlotCol_InlayText, ImVec4(1.0f, 1.0f, 1.0f, alpha0.x));
                            ImPlot::PlotText(std::to_string(scale0.x).c_str(), ((float*)gridBuffer)[54 + 0], ((float*)gridBuffer)[54 + 1], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();*/

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha1.x));
                            populateGridBuffer((float*)gridBuffer);
                            rotateOffsetBuffer2((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale1.x * window->scale.x, scale1.y * window->scale.y, scale1.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);

                            // y
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha0.y));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            rotateOffsetBuffer2((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale0.x * window->scale.x, scale0.y * window->scale.y, scale0.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha1.y));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            rotateOffsetBuffer2((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale1.x * window->scale.x, scale1.y * window->scale.y, scale1.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);

                            // z
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha0.z));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            gridY2Z((float*)gridBuffer);
                            rotateOffsetBuffer2((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale0.x * window->scale.x, scale0.y * window->scale.y, scale0.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);

                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, alpha1.z));
                            populateGridBuffer((float*)gridBuffer);
                            gridX2Y((float*)gridBuffer);
                            gridY2Z((float*)gridBuffer);
                            rotateOffsetBuffer2((float*)gridBuffer, 10 * 5 * 2, 3, 0, 1, 2, window->rotation.y, window->rotation.x, ImVec4(0, 0, 0, 0), ImVec4(scale1.x * window->scale.x, scale1.y * window->scale.y, scale1.z * window->scale.z, 0));
                            ImPlot::PlotLine(plotName.c_str(), &(((float*)gridBuffer)[0]), &(((float*)gridBuffer)[1]), 10 * 5 * 2, 0, 0, sizeof(float) * 3);
                        }
#endif
                    }

                    // PHASE DIAGRAM END
                    //printf("End phase\n");
                    ImPlot::EndPlot();
                }
                if (window->whiteBg) ImPlot::PopStyleColor(2);
                break;

                case Heatmap:
                    if (ImGui::BeginTable((plotName + "_table").c_str(), 2, ImGuiTableFlags_Reorderable, ImVec2(-1, 0)))
                    {
                        int heatStride = window->stride;
                        //if (autofitHeatmap) axisFlags |= ImPlotAxisFlags_AutoFit;

                        ImGui::TableSetupColumn(nullptr);
                        ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, 40.0f);
                        ImGui::TableNextRow();

                        numb min = 0.0f;
                        numb max = 0.0f;

                        ImGui::TableSetColumnIndex(0);

                        if (window->whiteBg) { ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); }
                        if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend, axisFlags, axisFlags))
                        {
                            plot = ImPlot::GetPlot(plotName.c_str());
                            plot->is3d = false;
                            plot->isHeatmapSelectionModeOn = window->isHeatmapSelectionModeOn;

                            if (mapBuffers[playedBufferIndex])
                            {
                                ImPlot::PushColormap(heatmapColorMap);
                                mapIndex = window->variables[0];
                                int xSize = kernel::MAP_DATA[mapIndex].xSize;
                                int ySize = kernel::MAP_DATA[mapIndex].ySize;

                                ImVec4 plotRect = ImVec4((float)plot->Axes[plot->CurrentX].Range.Min, (float)plot->Axes[plot->CurrentY].Range.Min,
                                    (float)plot->Axes[plot->CurrentX].Range.Max, (float)plot->Axes[plot->CurrentY].Range.Max); // minX, minY, maxX, maxY
                                //printf("%f %f %f %f\n", plotRect.x, plotRect.y, plotRect.z, plotRect.w);

                                numb valuesX = kernel::MAP_DATA[mapIndex].typeX == PARAMETER ? kernel::PARAM_VALUES[kernel::MAP_DATA[mapIndex].indexX] : kernel::MAP_DATA[mapIndex].typeX == VARIABLE ? kernel::VAR_VALUES[kernel::MAP_DATA[mapIndex].indexX] : 0;
                                numb valuesY = kernel::MAP_DATA[mapIndex].typeY == PARAMETER ? kernel::PARAM_VALUES[kernel::MAP_DATA[mapIndex].indexY] : kernel::MAP_DATA[mapIndex].typeY == VARIABLE ? kernel::VAR_VALUES[kernel::MAP_DATA[mapIndex].indexY] : 0;
                                numb stepsX = kernel::MAP_DATA[mapIndex].typeX == PARAMETER ? kernel::PARAM_STEPS[kernel::MAP_DATA[mapIndex].indexX] : kernel::MAP_DATA[mapIndex].typeX == VARIABLE ? kernel::VAR_STEPS[kernel::MAP_DATA[mapIndex].indexX] : 0;
                                numb stepsY = kernel::MAP_DATA[mapIndex].typeY == PARAMETER ? kernel::PARAM_STEPS[kernel::MAP_DATA[mapIndex].indexY] : kernel::MAP_DATA[mapIndex].typeY == VARIABLE ? kernel::VAR_STEPS[kernel::MAP_DATA[mapIndex].indexY] : 0;
                                numb maxX = kernel::MAP_DATA[mapIndex].typeX == PARAMETER ? kernel::PARAM_MAX[kernel::MAP_DATA[mapIndex].indexX] : kernel::MAP_DATA[mapIndex].typeX == VARIABLE ? kernel::VAR_MAX[kernel::MAP_DATA[mapIndex].indexX] : 0;
                                numb maxY = kernel::MAP_DATA[mapIndex].typeY == PARAMETER ? kernel::PARAM_MAX[kernel::MAP_DATA[mapIndex].indexY] : kernel::MAP_DATA[mapIndex].typeY == VARIABLE ? kernel::VAR_MAX[kernel::MAP_DATA[mapIndex].indexY] : 0;

                                int cutoffWidth;
                                int cutoffHeight;
                                int cutoffMinX;
                                int cutoffMinY;
                                int cutoffMaxX;
                                int cutoffMaxY;
                                numb valueMinX;
                                numb valueMinY;
                                numb valueMaxX;
                                numb valueMaxY;
                                int stepCountX;
                                int stepCountY;

                                if (!window->showActualDiapasons)
                                {
                                    // Step diapasons
                                    cutoffMinX = (int)floor(plotRect.x) - 1;    if (cutoffMinX < 0) cutoffMinX = 0;
                                    cutoffMinY = (int)floor(plotRect.y) - 1;    if (cutoffMinY < 0) cutoffMinY = 0;
                                    cutoffMaxX = (int)ceil(plotRect.z);         if (cutoffMaxX > xSize - 1) cutoffMaxX = xSize - 1;
                                    cutoffMaxY = (int)ceil(plotRect.w);         if (cutoffMaxY > ySize - 1) cutoffMaxY = ySize - 1;
                                }
                                else
                                {
                                    // Value diapasons
                                    stepCountX = calculateStepCount(valuesX, maxX, stepsX);
                                    stepCountY = calculateStepCount(valuesY, maxY, stepsY);

                                    cutoffMinX = stepFromValue(valuesX, stepsX, plotRect.x);    if (cutoffMinX < 0) cutoffMinX = 0;
                                    cutoffMinY = stepFromValue(valuesY, stepsY, plotRect.y);    if (cutoffMinY < 0) cutoffMinY = 0;
                                    cutoffMaxX = stepFromValue(valuesX, stepsX, plotRect.z);    if (cutoffMaxX > stepCountX - 1) cutoffMaxX = stepCountX - 1;
                                    cutoffMaxY = stepFromValue(valuesY, stepsY, plotRect.w);    if (cutoffMaxY > stepCountY - 1) cutoffMaxY = stepCountY - 1;

                                    valueMinX = calculateValue(valuesX, stepsX, cutoffMinX);
                                    valueMinY = calculateValue(valuesY, stepsY, cutoffMinY);
                                    valueMaxX = calculateValue(valuesX, stepsX, cutoffMaxX + 1);
                                    valueMaxY = calculateValue(valuesY, stepsY, cutoffMaxY + 1);
                                }

                                //printf("Cutoff: %i %i %i %i\n", cutoffMinX, cutoffMinY, cutoffMaxX, cutoffMaxY);

                                cutoffWidth = cutoffMaxX - cutoffMinX + 1;
                                cutoffHeight = cutoffMaxY - cutoffMinY + 1;

                                numb heatmapX1 = window->showActualDiapasons ? valuesX : 0;
                                numb heatmapX2 = window->showActualDiapasons ? maxX + stepsX : xSize;
                                numb heatmapY1 = window->showActualDiapasons ? valuesY : 0;
                                numb heatmapY2 = window->showActualDiapasons ? maxY + stepsY : ySize;

                                numb heatmapX1Cutoff = window->showActualDiapasons ? valueMinX : cutoffMinX;
                                numb heatmapX2Cutoff = window->showActualDiapasons ? valueMaxX : cutoffMaxX + 1;
                                numb heatmapY1Cutoff = window->showActualDiapasons ? valueMaxY : cutoffMaxY + 1;
                                numb heatmapY2Cutoff = window->showActualDiapasons ? valueMinY : cutoffMinY;

                                // Choosing configuration
                                if (plot->shiftClicked && plot->shiftClickLocation.x > 0.0)
                                {
                                    int stepX = 0;
                                    int stepY = 0;

                                    if (window->showActualDiapasons)
                                    {
                                        // Values
                                        stepX = stepFromValue(valuesX, stepsX, plot->shiftClickLocation.x);
                                        stepY = stepFromValue(valuesY, stepsY, plot->shiftClickLocation.y);
                                    }
                                    else
                                    {
                                        // Steps
                                        stepX = (int)floor(plot->shiftClickLocation.x);
                                        stepY = (int)floor(plot->shiftClickLocation.y);
                                    }

                                    enabledParticles = false;
                                    playingParticles = false;

                                    int rangingIndexX = rangingData[playedBufferIndex].indexOfRangingEntity(kernel::MAP_DATA[mapIndex].typeX == PARAMETER ? kernel::PARAM_NAMES[kernel::MAP_DATA[mapIndex].indexX] : kernel::MAP_DATA[mapIndex].typeX == VARIABLE ? kernel::VAR_NAMES[kernel::MAP_DATA[mapIndex].indexX] : "");
                                    int rangingIndexY = rangingData[playedBufferIndex].indexOfRangingEntity(kernel::MAP_DATA[mapIndex].typeY == PARAMETER ? kernel::PARAM_NAMES[kernel::MAP_DATA[mapIndex].indexY] : kernel::MAP_DATA[mapIndex].typeY == VARIABLE ? kernel::VAR_NAMES[kernel::MAP_DATA[mapIndex].indexY] : "");

                                    // If inside the heatmap
                                    if (stepX >= 0 && stepX < rangingData[playedBufferIndex].stepCount[rangingIndexX] && stepY >= 0 && stepY < rangingData[playedBufferIndex].stepCount[rangingIndexY])
                                    {
                                        if (rangingIndexX > -1)
                                        {
                                            rangingData[playedBufferIndex].currentStep[rangingIndexX] = stepX;
                                        }
                                        else
                                        {
                                            // TODO: if step is the ranging entity (not var or param)
                                            // ditto for y
                                        }

                                        if (rangingIndexY > -1)
                                        {
                                            rangingData[playedBufferIndex].currentStep[rangingIndexY] = stepY;
                                        }
                                    }
                                }

                                // Selecting new ranging
                                if (plot->shiftSelected)
                                {
                                    int stepX1 = 0;
                                    int stepY1 = 0;
                                    int stepX2 = 0;
                                    int stepY2 = 0;

                                    if (window->showActualDiapasons)
                                    {
                                        // Values
                                        stepX1 = stepFromValue(valuesX, stepsX, plot->shiftSelect1Location.x);
                                        stepY1 = stepFromValue(valuesY, stepsY, plot->shiftSelect1Location.y);
                                        stepX2 = stepFromValue(valuesX, stepsX, plot->shiftSelect2Location.x);
                                        stepY2 = stepFromValue(valuesY, stepsY, plot->shiftSelect2Location.y);
                                    }
                                    else
                                    {
                                        // Steps
                                        stepX1 = (int)floor(plot->shiftSelect1Location.x);
                                        stepY1 = (int)floor(plot->shiftSelect1Location.y);
                                        stepX2 = (int)floor(plot->shiftSelect2Location.x);
                                        stepY2 = (int)floor(plot->shiftSelect2Location.y);
                                    }

                                    enabledParticles = false;
                                    playingParticles = false;

                                    int rangingIndexX = rangingData[playedBufferIndex].indexOfRangingEntity(kernel::MAP_DATA[mapIndex].typeX == PARAMETER ? kernel::PARAM_NAMES[kernel::MAP_DATA[mapIndex].indexX] : kernel::MAP_DATA[mapIndex].typeX == VARIABLE ? kernel::VAR_NAMES[kernel::MAP_DATA[mapIndex].indexX] : "");
                                    int rangingIndexY = rangingData[playedBufferIndex].indexOfRangingEntity(kernel::MAP_DATA[mapIndex].typeY == PARAMETER ? kernel::PARAM_NAMES[kernel::MAP_DATA[mapIndex].indexY] : kernel::MAP_DATA[mapIndex].typeY == VARIABLE ? kernel::VAR_NAMES[kernel::MAP_DATA[mapIndex].indexY] : "");

                                    // If inside the heatmap
                                    if (stepX1 >= 0 && stepX1 < rangingData[playedBufferIndex].stepCount[rangingIndexX] && stepY1 >= 0 && stepY1 < rangingData[playedBufferIndex].stepCount[rangingIndexY]
                                        && stepX2 >= 0 && stepX2 < rangingData[playedBufferIndex].stepCount[rangingIndexX] && stepY2 >= 0 && stepY2 < rangingData[playedBufferIndex].stepCount[rangingIndexY])
                                    {
                                        //printf("%i:%i - %i:%i\n", stepX1, stepY1, stepX2, stepY2);

                                        int indexX = kernel::MAP_DATA[mapIndex].indexX;
                                        int indexY = kernel::MAP_DATA[mapIndex].indexY;

                                        if (kernel::MAP_DATA[mapIndex].typeX == VARIABLE)
                                        {
                                            varNew.MIN[indexX] = calculateValue(kernel::VAR_VALUES[indexX], kernel::VAR_STEPS[indexX], stepX1);
                                            varNew.MAX[indexX] = calculateValue(kernel::VAR_VALUES[indexX], kernel::VAR_STEPS[indexX], stepX2);
                                            varNew.RANGING[indexX] = Linear;
                                        }
                                        else
                                        {
                                            paramNew.MIN[indexX] = calculateValue(kernel::PARAM_VALUES[indexX], kernel::PARAM_STEPS[indexX], stepX1);
                                            paramNew.MAX[indexX] = calculateValue(kernel::PARAM_VALUES[indexX], kernel::PARAM_STEPS[indexX], stepX2);
                                            paramNew.RANGING[indexX] = Linear;
                                        }

                                        if (kernel::MAP_DATA[mapIndex].typeY == VARIABLE)
                                        {
                                            varNew.MIN[indexY] = calculateValue(kernel::VAR_VALUES[indexY], kernel::VAR_STEPS[indexY], stepY1);
                                            varNew.MAX[indexY] = calculateValue(kernel::VAR_VALUES[indexY], kernel::VAR_STEPS[indexY], stepY2);
                                            varNew.RANGING[indexY] = Linear;
                                        }
                                        else
                                        {
                                            paramNew.MIN[indexY] = calculateValue(kernel::PARAM_VALUES[indexY], kernel::PARAM_STEPS[indexY], stepY1);
                                            paramNew.MAX[indexY] = calculateValue(kernel::PARAM_VALUES[indexY], kernel::PARAM_STEPS[indexY], stepY2);
                                            paramNew.RANGING[indexY] = Linear;
                                        }
                                    }
                                }

                                if (autofitHeatmap || toAutofit)
                                {
                                    plot->Axes[plot->CurrentX].Range.Min = heatmapX1;
                                    plot->Axes[plot->CurrentX].Range.Max = heatmapX2;
                                    plot->Axes[plot->CurrentY].Range.Min = heatmapY1;
                                    plot->Axes[plot->CurrentY].Range.Max = heatmapY2;
                                }

                                // Actual drawing of the heatmap
                                if (cutoffWidth > 0 && cutoffHeight > 0)
                                {
                                    numb* mapData = (numb*)(mapBuffers[playedBufferIndex]) + kernel::MAP_DATA[mapIndex].offset;

                                    getMinMax(mapData, xSize * ySize, &min, &max);

                                    void* cutoffHeatmap = new numb[cutoffHeight * cutoffWidth];

                                    cutoff2D(mapData, (numb*)cutoffHeatmap,
                                        xSize, ySize, cutoffMinX, cutoffMinY, cutoffMaxX, cutoffMaxY);

                                    void* compressedHeatmap = new numb[(int)ceil((numb)cutoffWidth / heatStride) * (int)ceil((numb)cutoffHeight / heatStride)];

                                    compress2D((numb*)cutoffHeatmap, (numb*)compressedHeatmap,
                                        cutoffWidth, cutoffHeight, heatStride);

                                    int rows = heatStride > 1 ? (int)ceil((numb)cutoffHeight / heatStride) : cutoffHeight;
                                    int cols = heatStride > 1 ? (int)ceil((numb)cutoffWidth / heatStride) : cutoffWidth;

                                    ImPlot::PlotHeatmap((std::string(kernel::VAR_NAMES[mapIndex]) + "##" + plotName + std::to_string(0)).c_str(),
                                        (numb*)compressedHeatmap, rows, cols, (double)min, (double)max, window->showHeatmapValues ? "%.3f" : nullptr,
                                        ImPlotPoint(heatmapX1Cutoff, heatmapY1Cutoff), ImPlotPoint(heatmapX2Cutoff, heatmapY2Cutoff)); // %3f

                                    delete[] cutoffHeatmap;
                                    delete[] compressedHeatmap;
                                }

                                ImPlot::PopColormap();
                            }

                            ImPlot::EndPlot();
                        }
                        if (window->whiteBg) ImPlot::PopStyleColor(2);

                        // Legend

                        ImGui::TableSetColumnIndex(1);

                        axisFlags |= ImPlotAxisFlags_NoDecorations;

                        if (ImPlot::BeginPlot((plotName + "_legend").c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_CanvasOnly | ImPlotFlags_NoFrame, axisFlags, axisFlags))
                        {
                            plot = ImPlot::GetPlot((plotName + "_legend").c_str());
                            plot->is3d = false;

                            if (mapBuffers[playedBufferIndex])
                            {
                                float* legendData = new float[1000];
                                for (int i = 0; i < 1000; i++) legendData[i] = (float)i;

                                ImPlot::PushColormap(heatmapColorMap);
                                mapIndex = window->variables[0];
                                int xSize = kernel::MAP_DATA[mapIndex].xSize;
                                int ySize = kernel::MAP_DATA[mapIndex].ySize;

                                plot->Axes[plot->CurrentX].Range.Min = 0;
                                plot->Axes[plot->CurrentY].Range.Min = 0;
                                plot->Axes[plot->CurrentX].Range.Max = 1;
                                plot->Axes[plot->CurrentY].Range.Max = 1000;

                                //numb* mapData = (float*)(mapBuffers[playedBufferIndex]) + kernel::MAP_DATA[mapIndex].offset;

                                ImPlot::PlotHeatmap((std::string(kernel::VAR_NAMES[mapIndex]) + "_legend##" + plotName + std::to_string(0) + "_legend").c_str(),
                                    legendData, 1000, 1, 0, 999, nullptr,
                                    ImPlotPoint(0.0f, 1000.0f),
                                    ImPlotPoint(1.0f, 0.0f)); // %3f

                                ImPlot::PlotText(to_string(min).c_str(), 0.0f, 0.0f, ImVec2(10, -40), ImPlotTextFlags_Vertical);
                                ImPlot::PlotText(to_string(max).c_str(), 0.0f, 1000.0f, ImVec2(10, 80), ImPlotTextFlags_Vertical);

                                ImPlot::PopColormap();

                                delete[] legendData;
                            }

                            ImPlot::EndPlot();
                        }

                        ImGui::EndTable();
                    }

                    break;
            }          

            ImGui::End();
        }

        //autofitAfterComputing = false;

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
