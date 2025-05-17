#include "imgui_main.hpp"

#include "gui/plotWindowMenu.h"
#include "gui/img_loading.h"
#include "gui/map_img.h"

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

std::vector<PlotWindow> plotWindows;
int uniqueIds = 0; // Unique window IDs

Computation computations[2];
int playedBufferIndex = 0; // Buffer currently shown
int bufferToFillIndex = 0; // Buffer to send computations to
std::vector<int> attributeValueIndices;

bool autoLoadNewParams = false;
Kernel kernelNew;

numb* dataBuffer = nullptr; // One variation local buffer
numb* particleBuffer = nullptr; // One step local buffer
//numb* mapBuffer = nullptr;

float axisBuffer[18]{}; // 3 axis, 2 points
float rulerBuffer[153]{}; // 1 axis, 5 * 10 + 1 points

int computedSteps = 0; // Step count for the current computation
bool autofitAfterComputing = false; // Temporary flag to autofit computed data
int currentTotalVariations = 0; // Current amount of variations, so we can compare and safely hot-swap the parameter values
bool executedOnLaunch = false; // Temporary flag to execute computations on launch if needed

bool enabledParticles = false; // Particles mode
bool playingParticles = false; // Playing animation
float particleSpeed = 5000.0f; // Steps per second
float particlePhase = 0.0f; // Animation frame cooldown
int particleStep = 0; // Current step of the computations to show
bool continuousComputingEnabled = true; // Continuously compute next batch of steps via double buffering
float dragChangeSpeed = 1.0f;
int bufferNo = 0;

bool selectParticleTab = false;
bool selectOrbitTab = true;

bool computeAfterShiftSelect = false;
bool autofitHeatmap;

// Temporary variables
int variation = 0;
int prevVariation = 0;
int stride = 1;
float frameTime; // In seconds
float timeElapsed = 0.0f; // Total time elapsed, in seconds
int maxNameLength;
bool anyChanged;
bool thisChanged;
bool popStyle;
ImGuiSliderFlags dragFlag;

// Colors
ImVec4 unsavedBackgroundColor = ImVec4(0.427f, 0.427f, 0.137f, 1.0f);
ImVec4 unsavedBackgroundColorHovered = ImVec4(0.427f * 1.3f, 0.427f * 1.3f, 0.137f * 1.3f, 1.0f);
ImVec4 unsavedBackgroundColorActive = ImVec4(0.427f * 1.5f, 0.427f * 1.5f, 0.137f * 1.5f, 1.0f);
ImVec4 disabledColor = ImVec4(0.137f * 0.5f, 0.271f * 0.5f, 0.427f * 0.5f, 1.0f);
ImVec4 disabledTextColor = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);
ImVec4 disabledBackgroundColor = ImVec4(0.137f * 0.35f, 0.271f * 0.35f, 0.427f * 0.35f, 1.0f);
ImVec4 xAxisColor = ImVec4(0.75f, 0.3f, 0.3f, 1.0f);
ImVec4 yAxisColor = ImVec4(0.33f, 0.67f, 0.4f, 1.0f);
ImVec4 zAxisColor = ImVec4(0.3f, 0.45f, 0.7f, 1.0f);

std::string rangingTypes[] = { "Fixed", "Step", "Linear", "Random", "Normal" };
std::string rangingDescriptions[] =
{
    "Single value",
    "Values from 'min' to 'max' (inclusive), separated by 'step'",
    "Specified amount of values between 'min' and 'max' (inclusive)",
    "Uniform random distribution of values between 'min' and 'max'",
    "Normal random distribution of values around 'mu' with standard deviation 'sigma'"
};

bool rangingWindowEnabled = true;
bool graphBuilderWindowEnabled = true;

// Repetitive stuff
#define PUSH_DISABLED_FRAME {ImGui::PushStyleColor(ImGuiCol_FrameBg, disabledBackgroundColor); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, disabledBackgroundColor); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, disabledBackgroundColor);}
#define PUSH_UNSAVED_FRAME  {ImGui::PushStyleColor(ImGuiCol_FrameBg, unsavedBackgroundColor); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, unsavedBackgroundColorActive); \
                            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, unsavedBackgroundColorHovered);}
#define POP_FRAME(n)        {ImGui::PopStyleColor(n);}
#define CLAMP01(x)          if (x < 0.0f) x = 0.0f; if (x > 1.0f) x = 1.0f;
#define TOOLTIP(text)       if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip(text);

void deleteBothBuffers()
{
    computations[0].Clear();
    computations[1].Clear();

    playedBufferIndex = 0;
    bufferToFillIndex = 0;
}

void resetTempBuffers(Computation* data)
{
    if (dataBuffer) delete[] dataBuffer;
    dataBuffer = new numb[(CUDA_kernel.steps + 1) * KERNEL.VAR_COUNT];

    if (particleBuffer) delete[] particleBuffer;
    particleBuffer = new numb[CUDA_marshal.totalVariations * KERNEL.VAR_COUNT];

    //if (mapBuffer) delete[] mapBuffer;
    //mapBuffer = new numb[CUDA_marshal.mapsSize];
}

std::string padString(std::string str, int length)
{
    std::string strPadded = str;
    for (int j = (int)str.length(); j < length; j++)
        strPadded += ' ';
    return strPadded;
}

// Initialize the Attribute Value Indeces vector for ranging
void initAVI()
{
    attributeValueIndices.clear();
    for (int i = 0; i < kernelNew.VAR_COUNT + kernelNew.PARAM_COUNT; i++) attributeValueIndices.push_back(0);
}

void computing();

int asyncComputation()
{
    computations[bufferToFillIndex].ready = false;

    bool isFirstBatch = computations[1 - bufferToFillIndex].marshal.trajectory == nullptr; // Is another buffer null, only true when computing for the first time
    computations[bufferToFillIndex].isFirst = isFirstBatch;

    int computationResult = compute(&(computations[bufferToFillIndex]));

    computedSteps = KERNEL.steps;

    if (isFirstBatch)
    {
        autofitAfterComputing = true;
        resetTempBuffers(&(computations[bufferToFillIndex]));
        initAVI();
    }

    computations[bufferToFillIndex].ready = true;
    for (int i = 0; i < plotWindows.size(); i++)
    {
        plotWindows[i].hmp.initClickedLocation = true;
        plotWindows[i].hmp.isHeatmapDirty = true;
    }

    if (continuousComputingEnabled) bufferToFillIndex = 1 - bufferToFillIndex;
    if (continuousComputingEnabled && bufferToFillIndex != playedBufferIndex)
    {
        computing();
    }

    return computationResult;
}

void computing()
{
    computations[bufferToFillIndex].future = std::async(asyncComputation);
}

// Windows configuration saving and loading
void saveWindows()
{
    std::ofstream configFileStream((KERNEL.name + ".config").c_str(), std::ios::out);

    for (PlotWindow w : plotWindows)
    {
        std::string exportString = w.ExportAsString();
        configFileStream.write(exportString.c_str(), exportString.length());
    }

    configFileStream.close();
}

void loadWindows()
{
    std::ifstream configFileStream((KERNEL.name + ".config").c_str(), std::ios::in);

    for (std::string line; getline(configFileStream, line); )
    {
        PlotWindow plotWindow = PlotWindow(uniqueIds++);
        plotWindow.ImportAsString(line);

        plotWindows.push_back(plotWindow);
    }

    configFileStream.close();
}

void terminateBuffers()
{
    if (computations[0].future.valid()) computations[0].future.wait();
    if (computations[1].future.valid()) computations[1].future.wait();
    deleteBothBuffers();
    if (dataBuffer != nullptr)      { delete[] dataBuffer;      dataBuffer = nullptr; }
    if (particleBuffer != nullptr)  { delete[] particleBuffer;  particleBuffer = nullptr; }
    //if (mapBuffer != nullptr)       { delete[] mapBuffer;  mapBuffer = nullptr; }

    executedOnLaunch = false;
    playedBufferIndex = bufferToFillIndex = 0;
}

void initializeKernel(bool needTerminate)
{
    if (needTerminate) terminateBuffers();

    plotWindows.clear();
    uniqueIds = 0;

    kernelNew.CopyFrom(&KERNEL);

    computations[0].Clear();
    computations[1].Clear();

    initAVI();

    particleStep = 0;
    computeAfterShiftSelect = false;
}

void computationStatus(bool comp0, bool comp1)
{
    if (comp0)
    {
        ImGui::Text("Computing buffer 0...");
        return;
    }
    
    if (comp1)
    {
        ImGui::Text("Computing buffer 1...");
    }
}

void switchPlayedBuffer()
{
    if (computations[1 - playedBufferIndex].ready)
    {
        playedBufferIndex = 1 - playedBufferIndex;
        particleStep = 0;
        bufferNo++;
        //printf("Switch occured and starting playing %i\n", playedBufferIndex);
        computing();
    }
    else
    {
        //ImGui::Text("Next buffer not ready for switching yet!");
    }
}

void removeHeatmapLimits()
{
    for (int i = 0; i < plotWindows.size(); i++)
    {
        plotWindows[i].hmp.areHeatmapLimitsDefined = false;
    }
}

void prepareAndCompute()
{
    if ((!computations[0].ready && computations[0].marshal.trajectory != nullptr) || (!computations[1].ready && computations[1].marshal.trajectory != nullptr))
    {
        printf("Preventing computing too fast!\n");
    }
    else
    {
        executedOnLaunch = true;
        computeAfterShiftSelect = false;
        bufferToFillIndex = 0;
        playedBufferIndex = 0;
        bufferNo = 0;
        particleStep = 0;
        deleteBothBuffers();
        removeHeatmapLimits();

        KERNEL.CopyFrom(&kernelNew);
        KERNEL.PrepareAttributes();
        KERNEL.AssessMapAttributes(&attributeValueIndices);
        KERNEL.MapsSetSizes();

        // TODO: All calc steps and stepcounts should be done beforehand, since ranging will be incorrect otherwise
        // Done I guess? I forgor
        //initAVI();
        computing();
    }
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
    ImPlot3D::CreateContext();

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
    std::set<int> selectedPlotVarsSet;
    int selectedPlotMap = 0;

    computations[0].marshal.trajectory = computations[1].marshal.trajectory = nullptr;
    computations[0].marshal.parameterVariations = computations[1].marshal.parameterVariations = nullptr;

    computations[0].index = 0;
    computations[1].index = 1;
    computations[0].otherMarshal = &(computations[1].marshal);
    computations[1].otherMarshal = &(computations[0].marshal);
    
    initializeKernel(false);

    try { loadWindows(); }
    catch (std::exception e) { printf(e.what()); }

    while (work)
    {
        IMGUI_WORK_BEGIN;

        timeElapsed += frameTime;
        float breath = (cosf(timeElapsed * 6.0f) + 1.0f) / 2.0f;
        float buttonBreathMult = 1.2f + breath * 0.8f;
        bool noComputedData = computations[0].marshal.trajectory == nullptr;

        if (particleStep > computedSteps) particleStep = computedSteps;

        // MAIN WINDOW
        {
            style.WindowMenuButtonPosition = ImGuiDir_Left;
            ImGui::Begin("CUDAynamics", &work);

            // Selecting kernel
            if (ImGui::BeginCombo("##selectingKernel", KERNEL.name.c_str()))
            {
                for (auto k : kernels)
                {
                    bool isSelected = k.first == selectedKernel;
                    ImGuiSelectableFlags selectableFlags = 0;
                    if (ImGui::Selectable(k.second.name.c_str(), isSelected, selectableFlags))
                    {
                        saveWindows();

                        selectedKernel = k.first;
                        initializeKernel(true);
                        playingParticles = false;

                        loadWindows();
                    }
                }

                ImGui::EndCombo();
            }

            // Parameters & Variables

            maxNameLength = 0;
            for (int i = 0; i < KERNEL.PARAM_COUNT; i++) if (KERNEL.parameters[i].name.length() > maxNameLength) maxNameLength = (int)KERNEL.parameters[i].name.length();
            for (int i = 0; i < KERNEL.VAR_COUNT; i++) if (KERNEL.variables[i].name.length() > maxNameLength) maxNameLength = (int)KERNEL.variables[i].name.length();

            anyChanged = false;
            thisChanged = false;
            popStyle = false;

            ImGui::SeparatorText("Variables");

            for (int i = 0; i < KERNEL.VAR_COUNT; i++)
            {
                listVariable(i);
            }

            ImGui::SeparatorText("Parameters");

            bool applicationProhibited = false;

            for (int i = 0; i < KERNEL.PARAM_COUNT; i++)
            {
                listParameter(i);
            }

            if (playingParticles && anyChanged)
            {
                if (autoLoadNewParams)
                {
                    KERNEL.CopyParameterValuesFrom(&kernelNew);
                }
                else
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
                        KERNEL.CopyParameterValuesFrom(&kernelNew);
                    }
                    if (applicationProhibited)
                    {
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();
                    }
                }
            }

            // Simulation

            ImGui::SeparatorText("Simulation");

            int tempTotalVariations = 1;
            for (int v = 0; v < KERNEL.VAR_COUNT; v++)      if (kernelNew.variables[v].TrueStepCount() > 1)      tempTotalVariations *= kernelNew.variables[v].stepCount;
            for (int p = 0; p < KERNEL.PARAM_COUNT; p++)    if (kernelNew.parameters[p].TrueStepCount() > 1)    tempTotalVariations *= kernelNew.parameters[p].stepCount;
            unsigned long long tempTotalVariationsLL = tempTotalVariations;
            unsigned long long varCountLL = KERNEL.VAR_COUNT;
            unsigned long long stepsNewLL = kernelNew.steps + 1;
            unsigned long long singleBufferNumberCount = ((tempTotalVariationsLL * varCountLL) * stepsNewLL);
            unsigned long long singleBufferNumbSize = singleBufferNumberCount * sizeof(numb);
            ImGui::Text(("Single trajectory memory: " + memoryString(singleBufferNumbSize) + " (" + std::to_string(singleBufferNumbSize) + " bytes)").c_str());

            frameTime = 1.0f / io.Framerate; ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

            // Steps
            ImGui::PushItemWidth(200.0f);
            if (playingParticles)
            {
                ImGui::PushStyleColor(ImGuiCol_FrameBg, disabledBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, disabledBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, disabledBackgroundColor);
            }
            popStyle = false;
            if (kernelNew.steps != KERNEL.steps)
            {
                anyChanged = true;
                PUSH_UNSAVED_FRAME;
                popStyle = true;
            }
            ImGui::InputInt("Steps", &(kernelNew.steps), 1, 1000, playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
            TOOLTIP("Amount of computed steps, the trajectory will be (1 + 'steps') steps long, including the initial state");
            if (popStyle) POP_FRAME(3);
            if (playingParticles)
            {
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
            }

            // Transient steps
            ImGui::PushItemWidth(200.0f);
            if (playingParticles)
            {
                ImGui::PushStyleColor(ImGuiCol_FrameBg, disabledBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, disabledBackgroundColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, disabledBackgroundColor);
            }
            popStyle = false;
            if (kernelNew.transientSteps != KERNEL.transientSteps)
            {
                anyChanged = true;
                PUSH_UNSAVED_FRAME;
                popStyle = true;
            }
            ImGui::InputInt("Transient steps", &(kernelNew.transientSteps), 1, 1000, playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
            TOOLTIP("Steps to skip, including the initial state");
            if (popStyle) POP_FRAME(3);
            if (playingParticles)
            {
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
            }

            popStyle = false;
            if (kernelNew.stepSize != KERNEL.stepSize)
            {
                anyChanged = true;
                PUSH_UNSAVED_FRAME;
                popStyle = true;
            }
            ImGui::InputFloat("Step size", &(kernelNew.stepSize), 0.0f, 0.0f, "%f");
            TOOLTIP("Step size of the simulation, h");
            ImGui::PopItemWidth();
            if (popStyle) POP_FRAME(3);

            variation = 0;

            ImGui::NewLine();

            //enabledParticles = true;
            bool tempParticlesMode = enabledParticles;
            if (ImGui::Checkbox("Orbits/Particles", &(tempParticlesMode)))
            {
                enabledParticles = !enabledParticles;
            }

            // PARTICLES MODE
            ImGui::PushItemWidth(200.0f);
            ImGui::DragFloat("Animation speed, steps/s", &(particleSpeed), 1.0f);
            TOOLTIP("Playback speed of the evolution in Particles mode");
            if (particleSpeed < 0.0f) particleSpeed = 0.0f;
            ImGui::PopItemWidth();

            if (computations[playedBufferIndex].timeElapsed > 0.0f)
            {
                float buffersPerSecond = 1000.0f / computations[playedBufferIndex].timeElapsed;
                int stepsPerSecond = (int)(computedSteps * buffersPerSecond);

                ImGui::SameLine();
                ImGui::Text(("(max " + std::to_string(stepsPerSecond) + " before stalling)").c_str());
                TOOLTIP("Predicted speed that allows for seamless playback");
            }

            ImGui::PushItemWidth(200.0f);
            ImGui::DragInt("##Animation step", &(particleStep), 1.0f, 0, KERNEL.steps);
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::Text(("Animation step" + (continuousComputingEnabled ? " (total step " + std::to_string(bufferNo * KERNEL.steps + particleStep) + ")" : "")).c_str());

            if (ImGui::Button("Reset to step 0"))
            {
                particleStep = 0;
            }

            if (anyChanged)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
                PUSH_DISABLED_FRAME;
            }

            ImGui::Spacing();

            if (ImGui::Button(playingParticles ? "PAUSE" : "PLAY") && !anyChanged)
            {
                if (computations[0].ready || playingParticles)
                {
                    playingParticles = !playingParticles;
                    kernelNew.CopyFrom(&KERNEL);
                }

                if (!playingParticles)
                {
                    KERNEL.CopyFrom(&kernelNew);
                }
            }
            if (anyChanged) POP_FRAME(4);

            /*bool tempContinuous = continuousComputingEnabled;
            if (ImGui::Checkbox("Continuous computing", &(tempContinuous)))
            {
                // Flags of having buffers computed, to not interrupt computations in progress when switching
                bool noncont = !continuousComputingEnabled && computations[0].ready;
                bool cont = continuousComputingEnabled && computations[0].ready && computations[1].ready;

                if (noComputedData || noncont || cont)
                {
                    continuousComputingEnabled = !continuousComputingEnabled;

                    bufferNo = 0;
                    deleteBothBuffers();
                    playingParticles = false;
                    particleStep = 0;
                }
            }
            TOOLTIP("Keep computing next buffers for continuous playback");*/

            // PARTICLES MODE
            if (playingParticles/* && enabledParticles*/)
            {
                particlePhase += frameTime * particleSpeed;
                int passedSteps = (int)floor(particlePhase);
                particlePhase -= (float)passedSteps;

                particleStep += passedSteps;
                if (particleStep > KERNEL.steps) // Reached the end of animation
                {
                    if (continuousComputingEnabled)
                    {
                        // Starting from another buffer

                        switchPlayedBuffer();
                    }
                    else
                    {
                        // Stopping
                        particleStep = KERNEL.steps;
                        playingParticles = false;
                    }
                }
            }

            // Auto-loading
            if (playingParticles)
            {
                bool tempAutoLoadNewParams = autoLoadNewParams;
                if (ImGui::Checkbox("Apply parameter changes automatically", &(tempAutoLoadNewParams)))
                {
                    autoLoadNewParams = !autoLoadNewParams;
                    if (autoLoadNewParams) kernelNew.CopyParameterValuesFrom(&KERNEL);
                    else KERNEL.CopyParameterValuesFrom(&kernelNew);
                }
                TOOLTIP("Automatically applies new parameter values to the new buffers mid-playback");
            }

            ImGui::PushItemWidth(200.0f);
            ImGui::InputFloat("Value drag speed", &(dragChangeSpeed));
            TOOLTIP("Drag speed of attribute values, allows for precise automatic parameter setting");

            //if (ImGui::BeginTabItem("Orbit Mode", NULL, selectOrbitTab ? ImGuiTabItemFlags_SetSelected : 0))
            ImGui::NewLine();

            // RANGING, ORBIT MODE
            if (computations[playedBufferIndex].ready)
            {
                for (int i = 0; i < KERNEL.VAR_COUNT + KERNEL.PARAM_COUNT; i++)
                {
                    bool isVar = i < KERNEL.VAR_COUNT;
                    Attribute* attr = isVar ? &(computations[playedBufferIndex].marshal.kernel.variables[i]) : &(computations[playedBufferIndex].marshal.kernel.parameters[i - KERNEL.VAR_COUNT]);
                    Attribute* kernelNewAttr = isVar ? &(kernelNew.variables[i]) : &(kernelNew.parameters[i - KERNEL.VAR_COUNT]);

                    if (attr->TrueStepCount() == 1) continue;

                    ImGui::Text(padString(attr->name, maxNameLength).c_str()); ImGui::SameLine();
                    int index = attributeValueIndices[i];
                    ImGui::PushItemWidth(150.0f);
                    ImGui::SliderInt(("##RangingNo_" + std::to_string(i)).c_str(), &index, 0, attr->stepCount - 1, "Step: %d");
                    ImGui::PopItemWidth();
                    attributeValueIndices[i] = index;
                    ImGui::SameLine(); ImGui::Text(("Value: " + std::to_string(calculateValue(attr->min, attr->step, index))).c_str());
                }

                steps2Variation(&variation, &(attributeValueIndices.data()[0]), &KERNEL);
            }

            if (ImGui::Button("Next buffer"))
            {
                switchPlayedBuffer();
            }

            selectParticleTab = selectOrbitTab = false;

            // COMMON
            // default button color is 0.137 0.271 0.427
            bool playBreath = noComputedData || (anyChanged && (!playingParticles || !enabledParticles));
            if (playBreath)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.137f * buttonBreathMult, 0.271f * buttonBreathMult, 0.427f * buttonBreathMult, 1.0f));

            bool computation0InProgress = !computations[0].ready && computations[0].marshal.trajectory != nullptr;
            bool computation1InProgress = !computations[1].ready && computations[1].marshal.trajectory != nullptr;

            if (ImGui::Button("= COMPUTE =") || (KERNEL.executeOnLaunch && !executedOnLaunch) || computeAfterShiftSelect)
            {
                prepareAndCompute();
            }
            if (playBreath) ImGui::PopStyleColor();
            if (!playingParticles) computationStatus(computation0InProgress, computation1InProgress);

            // COMMON
            if (anyChanged && !autoLoadNewParams)
            {
                if (ImGui::Button("Reset changed values"))
                {
                    kernelNew.CopyFrom(&KERNEL);
                }
            }

            ImGui::End();

            // Graph Builder

            if (graphBuilderWindowEnabled)
            {
                ImGui::Begin("Graph Builder", &graphBuilderWindowEnabled);

                // Type
                std::string plottypes[] = { "Time series", "3D Phase diagram", "2D Phase diagram", "Orbit diagram", "Heatmap" };
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
                        for (int v = 0; v < KERNEL.VAR_COUNT; v++)
                        {
                            bool isSelected = selectedPlotVarsSet.find(v) != selectedPlotVarsSet.end();
                            ImGuiSelectableFlags selectableFlags = 0;

                            if (isSelected) selectableFlags = ImGuiSelectableFlags_Disabled;
                            if (ImGui::Selectable(KERNEL.variables[v].name.c_str())) selectedPlotVarsSet.insert(v);
                        }

                        ImGui::EndCombo();
                    }

                    // Variable list

                    for (const int v : selectedPlotVarsSet)
                    {
                        if (ImGui::Button(("x##" + std::to_string(v)).c_str()))
                        {
                            selectedPlotVarsSet.erase(v);
                            break; // temporary workaround (hahaha)
                            // What workaround? Why was it temporary?
                            // I forgor what was the original problem and I'm too afraid to find out what'll happen if I remove this workaround :skull:
                        }
                        ImGui::SameLine();
                        ImGui::Text(("- " + KERNEL.variables[v].name).c_str());
                    }

                    break;

                case Phase:
                    ImGui::PushItemWidth(150.0f);
                    for (int sv = 0; sv < 3; sv++)
                    {
                        ImGui::Text(("Variable " + variablexyz[sv]).c_str());
                        ImGui::SameLine();
                        if (ImGui::BeginCombo(("##Plot builder var " + std::to_string(sv + 1)).c_str(), selectedPlotVars[sv] > -1 ? KERNEL.variables[selectedPlotVars[sv]].name.c_str() : "-"))
                        {
                            for (int v = (sv > 0 ? -1 : 0); v < KERNEL.VAR_COUNT; v++)
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
                                    if (ImGui::Selectable(v > -1 ? KERNEL.variables[v].name.c_str() : "-", isSelected, selectableFlags)) selectedPlotVars[sv] = v;
                                }
                            }
                            ImGui::EndCombo();
                        }
                    }
                    ImGui::PopItemWidth();
                    break;

                case Phase2D:
                    ImGui::PushItemWidth(150.0f);
                    for (int sv = 0; sv < 2; sv++)
                    {
                        ImGui::Text(("Variable " + variablexyz[sv]).c_str());
                        ImGui::SameLine();
                        if (ImGui::BeginCombo(("##Plot builder var " + std::to_string(sv + 1)).c_str(), selectedPlotVars[sv] > -1 ? KERNEL.variables[selectedPlotVars[sv]].name.c_str() : "-"))
                        {
                            for (int v = (sv > 0 ? -1 : 0); v < KERNEL.VAR_COUNT; v++)
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
                                    if (ImGui::Selectable(v > -1 ? KERNEL.variables[v].name.c_str() : "-", isSelected, selectableFlags)) selectedPlotVars[sv] = v;
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
                    if (KERNEL.MAP_COUNT > 0)
                    {
                        ImGui::PushItemWidth(150.0f);

                        ImGui::Text("Index");
                        ImGui::SameLine();
                        /*if (ImGui::BeginCombo("##Plot builder map index selection", kernel::MAP_NAMES[selectedPlotMap]))
                        {
                            for (int m = 0; m < KERNEL.MAP_COUNT; m++)
                            {
                                bool isSelected = selectedPlotMap == m;
                                ImGuiSelectableFlags selectableFlags = 0;

                                if (selectedPlotMap == m) selectableFlags = ImGuiSelectableFlags_Disabled;
                                if (ImGui::Selectable(kernel::MAP_NAMES[m], isSelected, selectableFlags)) selectedPlotMap = m;
                            }
                            ImGui::EndCombo();
                        }*/ // TODO

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
                    if (plotType == Phase2D) plotWindow.AssignVariables(selectedPlotVars);
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
            ImGui::Begin(windowName.c_str(), &(window->active), ImGuiWindowFlags_MenuBar);

            autofitHeatmap = false;

            // Menu
            plotWindowMenu(window);

            // Plot variables
            /*if (ImGui::BeginCombo(("##" + windowName + "_plotSettings").c_str(), "Plot settings"))
            {
                if (window->type == Phase || window->type == Series)
                {
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

                ImGui::EndCombo();
            }*/

            // Heatmap axes
            if (window->type == Heatmap)
            {
                int prevIndexX = window->hmp.indexX;
                int prevIndexY = window->hmp.indexY;
                int prevTypeX = window->hmp.typeX;
                int prevTypeY = window->hmp.typeY;

                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.485f);
                if (ImGui::BeginCombo(("##" + windowName + "_axisX").c_str(),
                    window->hmp.typeX == VARIABLE ? KERNEL.variables[window->hmp.indexX].name.c_str() : KERNEL.parameters[window->hmp.indexX].name.c_str(), 0))
                {
                    for (int v = 0; v < KERNEL.VAR_COUNT; v++)
                    {
                        if (ImGui::Selectable(KERNEL.variables[v].name.c_str()))
                        {
                            window->hmp.indexX = v;
                            window->hmp.typeX = VARIABLE;
                        }
                    }

                    for (int p = 0; p < KERNEL.PARAM_COUNT; p++)
                    {
                        if (ImGui::Selectable(KERNEL.parameters[p].name.c_str()))
                        {
                            window->hmp.indexX = p;
                            window->hmp.typeX = PARAMETER;
                        }
                    }

                    ImGui::EndCombo();
                }

                ImGui::SameLine();

                if (ImGui::BeginCombo(("##" + windowName + "_axisY").c_str(),
                    window->hmp.typeY == VARIABLE ? KERNEL.variables[window->hmp.indexY].name.c_str() : KERNEL.parameters[window->hmp.indexY].name.c_str(), 0))
                {
                    for (int v = 0; v < KERNEL.VAR_COUNT; v++)
                    {
                        if (ImGui::Selectable(KERNEL.variables[v].name.c_str()))
                        {
                            window->hmp.indexY = v;
                            window->hmp.typeY = VARIABLE;
                        }
                    }

                    for (int p = 0; p < KERNEL.PARAM_COUNT; p++)
                    {
                        if (ImGui::Selectable(KERNEL.parameters[p].name.c_str()))
                        {
                            window->hmp.indexY = p;
                            window->hmp.typeY = PARAMETER;
                        }
                    }

                    ImGui::EndCombo();
                }

                ImGui::PopItemWidth();

                if (prevIndexX != window->hmp.indexX || prevIndexY != window->hmp.indexY || prevTypeX != window->hmp.typeX || prevTypeY != window->hmp.typeY)
                {
                    window->hmp.areValuesDirty = true;
                    window->hmp.areHeatmapLimitsDefined = false; // Triggers isHeatmapDirty too
                    //window->hmp.isHeatmapDirty = true;
                }
            }

            // Common variables
            ImPlotAxisFlags axisFlags = (toAutofit ? ImPlotAxisFlags_AutoFit : 0);
            ImPlotPlot* plot;
            ImPlot3DPlot* plot3d;
            int mapIndex;
            ImPlotColormap heatmapColorMap = !window->hmp.grayscaleHeatmap ? ImPlotColormap_Jet : ImPlotColormap_Greys;
            ImVec4 rotationEuler;
            ImVec4 rotationEulerEditable, rotationEulerBeforeEdit;

            switch (window->type)
            {
            case Series:

                if (window->whiteBg) ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle, axisFlags, axisFlags))
                {
                    plot = ImPlot::GetPlot(plotName.c_str());

                    plot->is3d = false;

                    if (computations[playedBufferIndex].ready)
                    {
                        int variationSize = KERNEL.VAR_COUNT * (computedSteps + 1);

                        void* computedVariation = (numb*)(computations[playedBufferIndex].marshal.trajectory) + (variationSize * variation);
                        memcpy(dataBuffer, computedVariation, variationSize * sizeof(numb));

                        for (int v = 0; v < window->variableCount; v++)
                        {
                            ImPlot::PlotLine((KERNEL.variables[window->variables[v]].name + "##" + plotName + std::to_string(v)).c_str(),
                                &((dataBuffer)[window->variables[v]]), computedSteps + 1, 1.0f, 0.0f, ImPlotLineFlags_None, 0, sizeof(numb) * KERNEL.VAR_COUNT);
                        }
                    }

                    ImPlot::EndPlot();
                }
                if (window->whiteBg) ImPlot::PopStyleColor();

                break;

            case Phase:
                // PHASE DIAGRAM
                rotationEuler = ToEulerAngles(window->quatRot);
                if (isnan(rotationEuler.x))
                {
                    window->quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
                    rotationEuler = ToEulerAngles(window->quatRot);
                }

                rotationEulerEditable = ImVec4(rotationEuler);
                rotationEulerEditable.x /= DEG2RAD;
                rotationEulerEditable.y /= DEG2RAD;
                rotationEulerEditable.z /= DEG2RAD;
                rotationEulerBeforeEdit = ImVec4(rotationEulerEditable);

                rotationEulerEditable.x += window->autorotate.x * frameTime;
                rotationEulerEditable.y += window->autorotate.y * frameTime;
                rotationEulerEditable.z += window->autorotate.z * frameTime;

                if (!window->isImplot3d)
                {
                    ImGui::DragFloat3("Rotation", (float*)&rotationEulerEditable, 1.0f);
                    ImGui::DragFloat3("Automatic rotation", (float*)&window->autorotate, 0.1f);
                    ImGui::DragFloat3("Offset", (float*)&window->offset, 0.01f);
                    ImGui::DragFloat3("Scale", (float*)&window->scale, 0.01f);
                }

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
                    quatEditable = quaternion::normalize(quatEditable);

                    window->quatRot.w = quatEditable.a();
                    window->quatRot.x = quatEditable.b();
                    window->quatRot.y = quatEditable.c();
                    window->quatRot.z = quatEditable.d();
                }

                axisFlags |= ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels;

                if (window->whiteBg) { ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); }

                bool isPlotBegan;
                if (window->isImplot3d)
                    isPlotBegan = ImPlot3D::BeginPlot(plotName.c_str(), ImVec2(-1, -1), ImPlot3DFlags_NoLegend | ImPlot3DFlags_NoTitle | ImPlot3DFlags_NoClip);
                else
                    isPlotBegan = ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, axisFlags, axisFlags);

                if (isPlotBegan)
                {
                    float plotRangeSize;

                    if (window->isImplot3d)
                        plot3d = ImPlot3D::GetCurrentPlot();
                    else
                    {
                        plot = ImPlot::GetCurrentPlot();

                        plotRangeSize = ((float)plot->Axes[ImAxis_X1].Range.Max - (float)plot->Axes[ImAxis_X1].Range.Min);

                        if (!computations[playedBufferIndex].ready)
                        {
                            plotRangeSize = 10.0f;
                            plot->dataMin = ImVec2(-10.0f, -10.0f);
                            plot->dataMax = ImVec2(10.0f, 10.0f);
                        }
                    }

                    float deltax = -window->deltarotation.x;
                    float deltay = -window->deltarotation.y;

                    window->deltarotation.x = 0;
                    window->deltarotation.y = 0;

                    if (!window->isImplot3d)
                    {
                        plot->is3d = true;
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
                        rotationEuler = ToEulerAngles(window->quatRot);

                        populateAxisBuffer(axisBuffer, plotRangeSize / 10.0f, plotRangeSize / 10.0f, plotRangeSize / 10.0f);
                        rotateOffsetBuffer(axisBuffer, 6, 3, 0, 1, 2, rotationEuler, ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 0));

                        // Axis
                        if (window->showAxis)
                        {
                            ImPlot::SetNextLineStyle(xAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[0]), &(axisBuffer[1]), 2, 0, 0, sizeof(float) * 3);
                            ImPlot::SetNextLineStyle(yAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[6]), &(axisBuffer[7]), 2, 0, 0, sizeof(float) * 3);
                            ImPlot::SetNextLineStyle(zAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[12]), &(axisBuffer[13]), 2, 0, 0, sizeof(float) * 3);
                        }

                        // Axis names
                        if (window->showAxisNames)
                        {
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, xAxisColor);
                            ImPlot::PlotText(KERNEL.variables[window->variables[0]].name.c_str(), axisBuffer[0], axisBuffer[1], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, yAxisColor);
                            ImPlot::PlotText(KERNEL.variables[window->variables[1]].name.c_str(), axisBuffer[6], axisBuffer[7], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, zAxisColor);
                            ImPlot::PlotText(KERNEL.variables[window->variables[2]].name.c_str(), axisBuffer[12], axisBuffer[13], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                        }

                        // Ruler
                        if (window->showRuler)
                        {
                            ImVec4 scale(plotRangeSize / window->scale.x, plotRangeSize / window->scale.y, plotRangeSize / window->scale.z, 0);
                            ImVec4 scaleLog(floorf(log10f(scale.x)), floorf(log10f(scale.y)), floorf(log10f(scale.z)), 0);
                            ImVec4 scale0(powf(10, scaleLog.x - 1), powf(10, scaleLog.y - 1), powf(10, scaleLog.z - 1), 0);
                            ImVec4 scale1(powf(10, scaleLog.x), powf(10, scaleLog.y), powf(10, scaleLog.z), 0);
                            ImVec4 scaleInterp(log10f(scale.x) - scaleLog.x, log10f(scale.y) - scaleLog.y, log10f(scale.z) - scaleLog.z, 0);

                            ImVec4 alpha0((1.0f - scaleInterp.x) * window->rulerAlpha, (1.0f - scaleInterp.y) * window->rulerAlpha, (1.0f - scaleInterp.z) * window->rulerAlpha, 0);
                            ImVec4 alpha1(scaleInterp.x * window->rulerAlpha, scaleInterp.y * window->rulerAlpha, scaleInterp.z * window->rulerAlpha, 0);

#define DRAW_RULER_PART(colorR, colorG, colorB, alpha, scale, scaleStr, dim) ImPlot::SetNextLineStyle(ImVec4(colorR, colorG, colorB, alpha)); \
                                populateRulerBuffer(rulerBuffer, scale, dim); \
                                rotateOffsetBuffer(rulerBuffer, 51, 3, 0, 1, 2, rotationEuler, \
                                    ImVec4(0, 0, 0, 0), ImVec4(scale, scale, scale, 0)); \
                                ImPlot::PlotLine(plotName.c_str(), &(rulerBuffer[0]), &(rulerBuffer[1]), 51, 0, 0, sizeof(float) * 3); \
                                ImPlot::PushStyleColor(ImPlotCol_InlayText, ImVec4(colorR, colorG, colorB, alpha)); \
                                ImPlot::PlotText(scaleString(scaleStr * 10.0f).c_str(), rulerBuffer[150 + 0], rulerBuffer[150 + 1], ImVec2(0.0f, 0.0f)); \
                                ImPlot::PopStyleColor();

                            DRAW_RULER_PART(xAxisColor.x, xAxisColor.y, xAxisColor.z, alpha0.x, scale0.x * window->scale.x, scale0.x, 0);
                            DRAW_RULER_PART(xAxisColor.x, xAxisColor.y, xAxisColor.z, alpha1.x, scale1.x * window->scale.x, scale1.x, 0);

                            DRAW_RULER_PART(yAxisColor.x, yAxisColor.y, yAxisColor.z, alpha0.y, scale0.y * window->scale.y, scale0.y, 1);
                            DRAW_RULER_PART(yAxisColor.x, yAxisColor.y, yAxisColor.z, alpha1.y, scale1.y * window->scale.y, scale1.y, 1);

                            DRAW_RULER_PART(zAxisColor.x, zAxisColor.y, zAxisColor.z, alpha0.z, scale0.z * window->scale.z, scale0.z, 2);
                            DRAW_RULER_PART(zAxisColor.x, zAxisColor.y, zAxisColor.z, alpha1.z, scale1.z * window->scale.z, scale1.z, 2);
                        }
                    }

                    if (computations[playedBufferIndex].ready)
                    {
                        if (!enabledParticles) // Trajectory - one variation, all steps
                        {
                            numb* computedVariation = computations[playedBufferIndex].marshal.trajectory + (computations[playedBufferIndex].marshal.variationSize * variation);

                            if (!window->isImplot3d)
                            {
                                memcpy(dataBuffer, computedVariation, computations[playedBufferIndex].marshal.variationSize * sizeof(numb));

                                rotateOffsetBuffer(dataBuffer, computedSteps + 1, KERNEL.VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                    rotationEuler, window->offset, window->scale);

                                getMinMax2D(dataBuffer, computedSteps + 1, &(plot->dataMin), &(plot->dataMax));
                                getMinMax2D(computedVariation, computedSteps + 1, &(plot->dataMin), &(plot->dataMax));

                                ImPlot::SetNextLineStyle(window->plotColor);
                                ImPlot::PlotLine(plotName.c_str(), &(dataBuffer[0]), &(dataBuffer[1]), computedSteps + 1, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
                            }
                            else
                            {
                                ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, ImVec4(0.07f, 0.07f, 0.07f, 1.0f));
                                ImPlot3D::SetNextLineStyle(window->plotColor);
                                ImPlot3D::SetupAxes(KERNEL.variables[window->variables[0]].name.c_str(), KERNEL.variables[window->variables[1]].name.c_str(), KERNEL.variables[window->variables[2]].name.c_str());
                                ImPlot3D::PlotLine(plotName.c_str(), &(computedVariation[window->variables[0]]), &(computedVariation[window->variables[1]]), &(computedVariation[window->variables[2]]),
                                    computedSteps + 1, 0, 0, sizeof(numb)* KERNEL.VAR_COUNT);
                                ImPlot3D::PopStyleColor(1);
                            }
                        }
                        else // Particles - all variations, one certain step
                        {
                            if (particleStep > KERNEL.steps) particleStep = KERNEL.steps;

                            int totalVariations = computations[playedBufferIndex].marshal.totalVariations;
                            int varCount = KERNEL.VAR_COUNT; // If you don't make this local, it increases the copying time by 30 times, tee-hee
                            int variationSize = computations[playedBufferIndex].marshal.variationSize;
                            numb* trajectory = computations[playedBufferIndex].marshal.trajectory;

                            for (int v = 0; v < totalVariations; v++)
                            {
                                for (int var = 0; var < varCount; var++)
                                    particleBuffer[v * varCount + var] = trajectory[(variationSize * v) + (varCount * particleStep) + var];
                            }

                            if (!window->isImplot3d)
                            {
                                getMinMax2D(particleBuffer, computations[playedBufferIndex].marshal.totalVariations, &(plot->dataMin), &(plot->dataMax));

                                rotateOffsetBuffer(particleBuffer, computations[playedBufferIndex].marshal.totalVariations, KERNEL.VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                    rotationEuler, window->offset, window->scale);

                                ImPlot::SetNextLineStyle(window->markerColor);
                                ImPlot::PlotScatter(plotName.c_str(), &((particleBuffer)[window->variables[0]]), &((particleBuffer)[window->variables[1]]),
                                    computations[playedBufferIndex].marshal.totalVariations, 0, 0, sizeof(numb)* KERNEL.VAR_COUNT);

                                /*if (plotWindows[1].hmp.lut != nullptr)
                                {
                                    for (int g = 0; g < 6; g++)
                                    {
                                        int lutsize = plotWindows[1].hmp.lutSizes[g];
                                        for (int v = 0; v < lutsize; v++)
                                        {
                                            for (int var = 0; var < varCount; var++)
                                                particleBuffer[v * varCount + var] = trajectory[(variationSize * plotWindows[1].hmp.lut[g][v]) + (varCount * particleStep) + var];
                                        }

                                        rotateOffsetBuffer(particleBuffer, lutsize, KERNEL.VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
                                            rotationEuler, window->offset, window->scale);

                                        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineSize);
                                        ImPlot::SetNextMarkerStyle(window->markerShape, window->markerSize);

                                        ImVec4 clr = ImVec4(0.5f, 0.0f, 0.5f, 1.0f);
                                        switch (g)
                                        {
                                        case 1: clr = ImVec4(0.0f, 0.0f, 1.0f, 1.0f); break;
                                        case 2: clr = ImVec4(0.0f, 1.0f, 1.0f, 1.0f); break;
                                        case 3: clr = ImVec4(0.0f, 1.0f, 1.0f, 1.0f); break;
                                        case 4: clr = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); break;
                                        case 5: clr = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); break;
                                        }

                                        ImPlot::SetNextLineStyle(clr);
                                        ImPlot::PlotScatter(plotName.c_str(), &((particleBuffer)[window->variables[0]]), &((particleBuffer)[window->variables[1]]),
                                            lutsize, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
                                    }
                                }*/
                            }
                            else
                            {
                                ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, ImVec4(0.07f, 0.07f, 0.07f, 1.0f));
                                ImPlot3D::SetNextLineStyle(window->markerColor);
                                ImPlot3D::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineSize);
                                ImPlot3D::SetNextMarkerStyle(window->markerShape, window->markerSize);
                                ImPlot3D::PlotScatter(plotName.c_str(), &((particleBuffer)[window->variables[0]]), &((particleBuffer)[window->variables[1]]), &((particleBuffer)[window->variables[2]]),
                                    computations[playedBufferIndex].marshal.totalVariations, 0, 0, sizeof(numb)* KERNEL.VAR_COUNT);
                                ImPlot3D::PopStyleColor(1);
                            }
                        }
                    }

                    // PHASE DIAGRAM END
                    if (window->isImplot3d)
                        ImPlot3D::EndPlot();
                    else
                        ImPlot::EndPlot();
                }
                if (window->whiteBg) ImPlot::PopStyleColor(2);
                break;

                case Phase2D:
                    // PHASE DIAGRAM 2D
                    rotationEuler = ToEulerAngles(window->quatRot);
                    if (isnan(rotationEuler.x))
                    {
                        window->quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
                        rotationEuler = ToEulerAngles(window->quatRot);
                    }

                    if (window->whiteBg) { ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); }

                    if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, axisFlags, axisFlags))
                    {
                        plot = ImPlot::GetPlot(plotName.c_str());

                        float plotRangeSize = ((float)plot->Axes[ImAxis_X1].Range.Max - (float)plot->Axes[ImAxis_X1].Range.Min);

                        if (!computations[playedBufferIndex].ready)
                        {
                            plotRangeSize = 10.0f;
                            plot->dataMin = ImVec2(-10.0f, -10.0f);
                            plot->dataMax = ImVec2(10.0f, 10.0f);
                        }

                        float deltax = -window->deltarotation.x;
                        float deltay = -window->deltarotation.y;

                        window->deltarotation.x = 0;
                        window->deltarotation.y = 0;

                        plot->is3d = false;
                        plot->deltax = &(window->deltarotation.x);
                        plot->deltay = &(window->deltarotation.y);

                        populateAxisBuffer(axisBuffer, plotRangeSize / 10.0f, plotRangeSize / 10.0f, plotRangeSize / 10.0f);

                        // Axis
                        if (window->showAxis)
                        {
                            ImPlot::SetNextLineStyle(xAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[0]), &(axisBuffer[1]), 2, 0, 0, sizeof(float) * 3);
                            ImPlot::SetNextLineStyle(yAxisColor);
                            ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[6]), &(axisBuffer[7]), 2, 0, 0, sizeof(float) * 3);
                        }

                        // Axis names
                        if (window->showAxisNames)
                        {
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, xAxisColor);
                            ImPlot::PlotText(KERNEL.variables[window->variables[0]].name.c_str(), axisBuffer[0], axisBuffer[1], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                            ImPlot::PushStyleColor(ImPlotCol_InlayText, yAxisColor);
                            ImPlot::PlotText(KERNEL.variables[window->variables[1]].name.c_str(), axisBuffer[6], axisBuffer[7], ImVec2(0.0f, 0.0f));
                            ImPlot::PopStyleColor();
                        }

                        if (computations[playedBufferIndex].ready)
                        {
                            int xIndex = window->variables[0];
                            int yIndex = window->variables[1];

                            if (!enabledParticles) // Trajectory - one variation, all steps
                            {
                                numb* computedVariation = computations[playedBufferIndex].marshal.trajectory + (computations[playedBufferIndex].marshal.variationSize * variation);

                                ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                                ImPlot::PlotLine(plotName.c_str(), &(computedVariation[0]), &(computedVariation[1]), computedSteps + 1, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
                            }
                            else // Particles - all variations, one certain step
                            {
                                if (particleStep > KERNEL.steps) particleStep = KERNEL.steps;

                                int totalVariations = computations[playedBufferIndex].marshal.totalVariations;
                                int varCount = KERNEL.VAR_COUNT; // If you don't make this local, it increases the copying time by 30 times, tee-hee
                                int variationSize = computations[playedBufferIndex].marshal.variationSize;
                                numb* trajectory = computations[playedBufferIndex].marshal.trajectory;

                                for (int v = 0; v < totalVariations; v++)
                                {
                                    for (int var = 0; var < varCount; var++)
                                        particleBuffer[v * varCount + var] = trajectory[(variationSize * v) + (varCount * particleStep) + var];
                                }

                                getMinMax2D(particleBuffer, computations[playedBufferIndex].marshal.totalVariations, &(plot->dataMin), &(plot->dataMax));

                                ImPlot::SetNextLineStyle(window->markerColor);
                                ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineSize);
                                ImPlot::SetNextMarkerStyle(window->markerShape, window->markerSize);
                                ImPlot::PlotScatter(plotName.c_str(), &((particleBuffer)[xIndex]), &((particleBuffer)[yIndex]), totalVariations, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
                            }
                        }

                        // PHASE DIAGRAM END
                        ImPlot::EndPlot();
                    }
                    if (window->whiteBg) ImPlot::PopStyleColor(2);
                    break;
                
                case Heatmap:
                    mapIndex = window->variables[0];
                    if (!KERNEL.mapDatas[mapIndex].toCompute) break;

                    Attribute* axisX = window->hmp.typeX == VARIABLE ? &(KERNEL.variables[window->hmp.indexX]) : &(KERNEL.parameters[window->hmp.indexX]);
                    Attribute* axisY = window->hmp.typeY == VARIABLE ? &(KERNEL.variables[window->hmp.indexY]) : &(KERNEL.parameters[window->hmp.indexY]);

                    bool axisXisRanging = axisX->TrueStepCount() > 1;
                    bool axisYisRanging = axisY->TrueStepCount() > 1;
                    bool sameAxis = axisX == axisY;

                    if (axisXisRanging && axisYisRanging && !sameAxis)
                    {
                        if (ImGui::BeginTable((plotName + "_table").c_str(), 2, ImGuiTableFlags_Reorderable, ImVec2(-1, 0)))
                        {
                            int heatStride = window->hmp.stride;
                            axisFlags = 0;

                            ImGui::TableSetupColumn(nullptr);
                            ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, 160.0f);
                            ImGui::TableNextRow();

                            numb min = 0.0f;
                            numb max = 0.0f;

                            ImGui::TableSetColumnIndex(0);
                            ImPlot::PushColormap(heatmapColorMap);
                            ImVec2 plotSize;

                            HeatmapSizing sizing;

                            if (window->whiteBg) { ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); }
                            if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend, axisFlags, axisFlags))
                            {
                                plot = ImPlot::GetPlot(plotName.c_str());
                                plot->is3d = false;
                                plot->isHeatmapSelectionModeOn = window->hmp.isHeatmapSelectionModeOn;

                                if (computations[playedBufferIndex].ready)
                                {
                                    sizing.loadPointers(&KERNEL, &(window->hmp));
                                    sizing.initValues();

                                    if (window->hmp.initClickedLocation)
                                    {
                                        window->hmp.lastClickedLocation = ImVec2(sizing.minX, sizing.minY);
                                        window->hmp.initClickedLocation = false;
                                    }

                                    if (window->hmp.showActualDiapasons)
                                    {
                                        // Values
                                        window->hmp.lastClickedLocation.x = valueFromStep(sizing.minX, sizing.stepX,
                                            attributeValueIndices[sizing.hmp->indexX + (sizing.hmp->typeX == VARIABLE ? 0 : KERNEL.VAR_COUNT)]);
                                        window->hmp.lastClickedLocation.y = valueFromStep(sizing.minY, sizing.stepY,
                                            attributeValueIndices[sizing.hmp->indexY + (sizing.hmp->typeY == VARIABLE ? 0 : KERNEL.VAR_COUNT)]);
                                    }
                                    else
                                    {
                                        // Steps
                                        window->hmp.lastClickedLocation.x = (float)attributeValueIndices[sizing.hmp->indexX + (sizing.hmp->typeX == VARIABLE ? 0 : KERNEL.VAR_COUNT)];
                                        window->hmp.lastClickedLocation.y = (float)attributeValueIndices[sizing.hmp->indexY + (sizing.hmp->typeY == VARIABLE ? 0 : KERNEL.VAR_COUNT)];
                                    }

                                    // Choosing configuration
                                    if (plot->shiftClicked && plot->shiftClickLocation.x != 0.0)
                                    {
                                        int stepX = 0;
                                        int stepY = 0;

                                        if (window->hmp.showActualDiapasons)
                                        {
                                            // Values
                                            stepX = stepFromValue(sizing.minX, sizing.stepX, plot->shiftClickLocation.x);
                                            stepY = stepFromValue(sizing.minY, sizing.stepY, plot->shiftClickLocation.y);
                                        }
                                        else
                                        {
                                            // Steps
                                            stepX = (int)floor(plot->shiftClickLocation.x);
                                            stepY = (int)floor(plot->shiftClickLocation.y);
                                        }

                                        //enabledParticles = false;
                                        playingParticles = false;

#define IGNOREOUTOFREACH    if (stepX < 0 || stepX >= (sizing.hmp->typeX == VARIABLE ? KERNEL.variables[sizing.hmp->indexX].TrueStepCount() : KERNEL.parameters[sizing.hmp->indexX].TrueStepCount())) break; \
                            if (stepY < 0 || stepY >= (sizing.hmp->typeY == VARIABLE ? KERNEL.variables[sizing.hmp->indexY].TrueStepCount() : KERNEL.parameters[sizing.hmp->indexY].TrueStepCount())) break;

                                        switch (sizing.hmp->typeX)
                                        {
                                        case VARIABLE:
                                            IGNOREOUTOFREACH;
                                            attributeValueIndices[sizing.hmp->indexX] = stepX;
                                            window->hmp.lastClickedLocation.x = plot->shiftClickLocation.x;
                                            break;
                                        case PARAMETER:
                                            IGNOREOUTOFREACH;
                                            attributeValueIndices[KERNEL.VAR_COUNT + sizing.hmp->indexX] = stepX;
                                            window->hmp.lastClickedLocation.x = plot->shiftClickLocation.x;
                                            break;
                                        }

                                        switch (sizing.hmp->typeY)
                                        {
                                        case VARIABLE:
                                            IGNOREOUTOFREACH;
                                            attributeValueIndices[sizing.hmp->indexY] = stepY;
                                            window->hmp.lastClickedLocation.y = plot->shiftClickLocation.y;
                                            break;
                                        case PARAMETER:
                                            IGNOREOUTOFREACH;
                                            attributeValueIndices[KERNEL.VAR_COUNT + sizing.hmp->indexY] = stepY;
                                            window->hmp.lastClickedLocation.y = plot->shiftClickLocation.y;
                                            break;
                                        }
                                    }

                                    if (plot->shiftSelected)
                                    {
                                        heatmapRangingSelection(window, plot, &sizing);
                                    }

                                    sizing.initCutoff((float)plot->Axes[plot->CurrentX].Range.Min, (float)plot->Axes[plot->CurrentY].Range.Min,
                                        (float)plot->Axes[plot->CurrentX].Range.Max, (float)plot->Axes[plot->CurrentY].Range.Max, window->hmp.showActualDiapasons);
                                    if (autofitHeatmap || toAutofit)
                                    {
                                        plot->Axes[plot->CurrentX].Range.Min = sizing.mapX1;
                                        plot->Axes[plot->CurrentX].Range.Max = sizing.mapX2;
                                        plot->Axes[plot->CurrentY].Range.Min = sizing.mapY1;
                                        plot->Axes[plot->CurrentY].Range.Max = sizing.mapY2;
                                    }

                                    // Drawing

                                    int mapSize = sizing.xSize * sizing.ySize;
                                    if (window->hmp.lastBufferSize != mapSize)
                                    {
                                        if (window->hmp.valueBuffer != nullptr) delete[] window->hmp.valueBuffer;
                                        if (window->hmp.pixelBuffer != nullptr) delete[] window->hmp.pixelBuffer;

                                        window->hmp.lastBufferSize = mapSize;

                                        window->hmp.valueBuffer = new numb[mapSize];
                                        window->hmp.pixelBuffer = new unsigned char[mapSize * 4];
                                        window->hmp.areValuesDirty = true;
                                    }

                                    if (variation != prevVariation) window->hmp.areValuesDirty = true;

                                    if (window->hmp.areValuesDirty)
                                    {
                                        extractMap(computations[playedBufferIndex].marshal.maps2, window->hmp.valueBuffer, &(attributeValueIndices.data()[0]),
                                            sizing.hmp->typeX == PARAMETER ? sizing.hmp->indexX + KERNEL.VAR_COUNT : sizing.hmp->indexX,
                                            sizing.hmp->typeY == PARAMETER ? sizing.hmp->indexY + KERNEL.VAR_COUNT : sizing.hmp->indexY,
                                            &KERNEL);
                                    }

                                    //numb* mapData = computations[playedBufferIndex].marshal.maps2 + sizing.map->offset;

                                    if (!window->hmp.areHeatmapLimitsDefined)
                                    {
                                        getMinMax(window->hmp.valueBuffer, sizing.xSize * sizing.ySize, &window->hmp.heatmapMin, &window->hmp.heatmapMax);
                                        window->hmp.areHeatmapLimitsDefined = true;
                                        window->hmp.isHeatmapDirty = true;
                                    }

                                    // TODO: Do not reload values when variating map axes (map values don't change anyway)
                                    if (variation != prevVariation)
                                    {
                                        window->hmp.isHeatmapDirty = true;
                                    }

                                    // Image init
                                    if (window->hmp.isHeatmapDirty)
                                    {
                                        MapToImg(window->hmp.valueBuffer, &(window->hmp.pixelBuffer), sizing.xSize, sizing.ySize, window->hmp.heatmapMin, window->hmp.heatmapMax);
                                        window->hmp.isHeatmapDirty = false;

                                        // WIP
                                        /*
                                        window->hmp.lut = new int* [6];
                                        for (int i = 0; i < 6; i++) window->hmp.lut[i] = new int[computations[playedBufferIndex].marshal.totalVariations];
                                        window->hmp.lutSizes = new int[6];

                                        setupLUT(window->hmp.valueBuffer, computations[playedBufferIndex].marshal.totalVariations, window->hmp.lut, window->hmp.lutSizes, 6, window->hmp.heatmapMin, window->hmp.heatmapMax);
                                        */
                                    }
                                    bool ret = LoadTextureFromRaw(&(window->hmp.pixelBuffer), sizing.xSize, sizing.ySize, (ID3D11ShaderResourceView**)&(window->hmp.myTexture), g_pd3dDevice);
                                    IM_ASSERT(ret);

                                    ImPlotPoint from = window->hmp.showActualDiapasons ? ImPlotPoint(sizing.minX, sizing.maxY + sizing.stepY) : ImPlotPoint(0, sizing.ySize);
                                    ImPlotPoint to = window->hmp.showActualDiapasons ? ImPlotPoint(sizing.maxX + sizing.stepX, sizing.minY) : ImPlotPoint(sizing.xSize, 0);

                                    ImPlot::SetupAxes(sizing.hmp->typeX == PARAMETER ? KERNEL.parameters[sizing.hmp->indexX].name.c_str() : KERNEL.variables[sizing.hmp->indexX].name.c_str(),
                                        sizing.hmp->typeY == PARAMETER ? KERNEL.parameters[sizing.hmp->indexY].name.c_str() : KERNEL.variables[sizing.hmp->indexY].name.c_str());

                                    ImPlot::PlotImage(("Map " + std::to_string(mapIndex) + "##" + plotName + std::to_string(0)).c_str(), (ImTextureID)(window->hmp.myTexture),
                                        from, to, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

                                    // Value labels
                                    if (sizing.cutWidth > 0 && sizing.cutHeight > 0)
                                    {
                                        if (window->hmp.showHeatmapValues)
                                        {
                                            void* cutoffHeatmap = new numb[sizing.cutHeight * sizing.cutWidth];

                                            cutoff2D(window->hmp.valueBuffer, (numb*)cutoffHeatmap,
                                                sizing.xSize, sizing.ySize, sizing.cutMinX, sizing.cutMinY, sizing.cutMaxX, sizing.cutMaxY);

                                            void* compressedHeatmap = new numb[(int)ceil((float)sizing.cutWidth / heatStride) * (int)ceil((float)sizing.cutHeight / heatStride)];

                                            compress2D((numb*)cutoffHeatmap, (numb*)compressedHeatmap,
                                                sizing.cutWidth, sizing.cutHeight, heatStride);

                                            int rows = heatStride > 1 ? (int)ceil((float)sizing.cutHeight / heatStride) : sizing.cutHeight;
                                            int cols = heatStride > 1 ? (int)ceil((float)sizing.cutWidth / heatStride) : sizing.cutWidth;

                                            ImPlot::PlotHeatmap(("MapLabels " + std::to_string(mapIndex) + "##" + plotName + std::to_string(0)).c_str(),
                                                (numb*)compressedHeatmap, rows, cols, -1234.0, 1.0, "%.3f",
                                                ImPlotPoint(sizing.mapX1Cut, sizing.mapY1Cut), ImPlotPoint(sizing.mapX2Cut, sizing.mapY2Cut));

                                            delete[] cutoffHeatmap;
                                            delete[] compressedHeatmap;
                                        }
                                    }

                                    double valueX = (double)window->hmp.lastClickedLocation.x + (window->hmp.showActualDiapasons ? sizing.stepX * 0.5 : 0.5);
                                    double valueY = (double)window->hmp.lastClickedLocation.y + (window->hmp.showActualDiapasons ? sizing.stepY * 0.5 : 0.5);

                                    ImPlot::DragLineX(0, &valueX, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), 2.0f, ImPlotDragToolFlags_NoInputs);
                                    ImPlot::DragLineY(1, &valueY, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), 2.0f, ImPlotDragToolFlags_NoInputs);

                                    plotSize = ImPlot::GetPlotSize();
                                }

                                ImPlot::EndPlot();
                            }

                            // Table column should be here
                            ImGui::TableSetColumnIndex(1);

                            ImGui::SetNextItemWidth(120);
                            float minBefore = window->hmp.heatmapMin;
                            float maxBefore = window->hmp.heatmapMax;
                            ImGui::DragFloat("Max", &window->hmp.heatmapMax, 0.01f);
                            ImPlot::ColormapScale("##HeatScale", window->hmp.heatmapMin, window->hmp.heatmapMax, ImVec2(120, plotSize.y - 30.0f));
                            ImGui::SetNextItemWidth(120);
                            ImGui::DragFloat("Min", &window->hmp.heatmapMin, 0.01f);
                            ImPlot::PopColormap();

                            if (minBefore != window->hmp.heatmapMin || maxBefore != window->hmp.heatmapMax) window->hmp.isHeatmapDirty = true;

                            if (window->whiteBg) ImPlot::PopStyleColor(2);

                            ImGui::EndTable();
                        }// if (ImGui::BeginTable
                    } // if (!axisXisRanging || !axisYisRanging)
                    else
                    {
                        if (!axisXisRanging) ImGui::Text(("Axis " + axisX->name + " is fixed").c_str());
                        if (!axisYisRanging) ImGui::Text(("Axis " + axisY->name + " is fixed").c_str());
                        if (sameAxis) ImGui::Text("X and Y axis are the same");
                    }

                    break;
            }          

            ImGui::End();
        }

        // Rendering
        IMGUI_WORK_END;

        // Cleaning
        for (int i = 0; i < plotWindows.size(); i++)
        {
            if (plotWindows[i].hmp.myTexture != nullptr)
            {
                ((ID3D11ShaderResourceView*)plotWindows[i].hmp.myTexture)->Release();
                plotWindows[i].hmp.myTexture = nullptr;
            }
        }

        prevVariation = variation;
    }

    saveWindows();

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    terminateBuffers();
    for (int i = 0; i < plotWindows.size(); i++) if (plotWindows[i].hmp.pixelBuffer != nullptr) delete[] plotWindows[i].hmp.pixelBuffer;

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

// Non-ImGui functions

#define ATTR_BEGIN  ImGui::SameLine(); popStyle = false; if (isChanged && !autoLoadNewParams) { PUSH_UNSAVED_FRAME; popStyle = true; }
#define ATTR_END    ImGui::PopItemWidth(); if (popStyle) POP_FRAME(3);

void listAttrRanging(Attribute* attr, bool isChanged)
{
    ATTR_BEGIN;
    ImGui::PushItemWidth(120.0f);
    if (ImGui::BeginCombo(("##RANGING_" + attr->name).c_str(), (rangingTypes[attr->rangingType]).c_str()))
    {
        for (int r = 0; r < 5; r++)
        {
            bool isSelected = attr->rangingType == r;
            ImGuiSelectableFlags selectableFlags = 0;
            if (ImGui::Selectable(rangingTypes[r].c_str(), isSelected, selectableFlags) && !playingParticles)
            {
                attr->rangingType = (RangingType)r;
            }
            TOOLTIP(rangingDescriptions[r].c_str());
        }

        ImGui::EndCombo();
    }
    ATTR_END;
}

void listAttrNumb(Attribute* attr, numb* field, std::string name, std::string inner, bool isChanged)
{
    ATTR_BEGIN;
    ImGui::PushItemWidth(200.0f);
    float var = (float)(*field);
    ImGui::DragFloat(("##" + name + attr->name).c_str(), &var, dragChangeSpeed, 0.0f, 0.0f, (inner + "%f").c_str(), dragFlag);
    (*field) = (numb)var;
    ATTR_END;
}

void listAttrInt(Attribute* attr, int* field, std::string name, std::string inner, bool isChanged)
{
    ATTR_BEGIN;
    ImGui::PushItemWidth(200.0f);
    ImGui::DragInt(("##" + name + attr->name).c_str(), field, dragChangeSpeed, 0, 0, (inner + "%i").c_str(), dragFlag);
    ATTR_END;
}

void listVariable(int i)
{
    thisChanged = false;
    if (kernelNew.variables[i].IsDifferentFrom(&(KERNEL.variables[i]))) { anyChanged = true; thisChanged = true; }
    //if (thisChanged) varNew.recountSteps(i); // TODO

    ImGui::Text(padString(KERNEL.variables[i].name, maxNameLength).c_str());

    if (playingParticles)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
        PUSH_DISABLED_FRAME;
    }

    // Ranging
    listAttrRanging(&(kernelNew.variables[i]), kernelNew.variables[i].rangingType != KERNEL.variables[i].rangingType);

    dragFlag = !playingParticles ? 0 : ImGuiSliderFlags_ReadOnly;

    switch (kernelNew.variables[i].rangingType)
    {
    case None:
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].min), "", "", kernelNew.variables[i].min != KERNEL.variables[i].min);
        break;
    case Step:
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].min), "", "Min: ", kernelNew.variables[i].min != KERNEL.variables[i].min);
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].max), "MAX", "Max: ", kernelNew.variables[i].max != KERNEL.variables[i].max);
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].step), "STEP", "Step: ", kernelNew.variables[i].step != KERNEL.variables[i].step);
        kernelNew.variables[i].CalcStepCount();
        ImGui::SameLine(); ImGui::Text((std::to_string(kernelNew.variables[i].stepCount) + " steps").c_str());
        break;
    case Linear:
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].min), "", "Min: ", kernelNew.variables[i].min != KERNEL.variables[i].min);
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].max), "MAX", "Max: ", kernelNew.variables[i].max != KERNEL.variables[i].max);
        listAttrInt(&(kernelNew.variables[i]), &(kernelNew.variables[i].stepCount), "STEPCOUNT", "Count: ", kernelNew.variables[i].stepCount != KERNEL.variables[i].stepCount);
        kernelNew.variables[i].CalcStep();
        ImGui::SameLine(); ImGui::Text(("Step: " + (std::to_string(kernelNew.variables[i].step))).c_str());
        break;
    case UniformRandom:
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].mean), "MEAN", "Mean: ", kernelNew.variables[i].mean != KERNEL.variables[i].mean);
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].deviation), "DEV", "Dev: ", kernelNew.variables[i].deviation != KERNEL.variables[i].deviation);
        listAttrInt(&(kernelNew.variables[i]), &(kernelNew.variables[i].stepCount), "STEPCOUNT", "Count: ", kernelNew.variables[i].stepCount != KERNEL.variables[i].stepCount);
        break;
    case NormalRandom:
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].mean), "MU", "Mu: ", kernelNew.variables[i].mean != KERNEL.variables[i].mean);
        listAttrNumb(&(kernelNew.variables[i]), &(kernelNew.variables[i].deviation), "SIGMA", "Sigma: ", kernelNew.variables[i].deviation != KERNEL.variables[i].deviation);
        listAttrInt(&(kernelNew.variables[i]), &(kernelNew.variables[i].stepCount), "STEPCOUNT", "Count: ", kernelNew.variables[i].stepCount != KERNEL.variables[i].stepCount);
        break;
    }

    if (playingParticles)
    {
        ImGui::PopStyleColor();
        POP_FRAME(3);
    }
}

void listParameter(int i)
{
    bool changeAllowed = kernelNew.parameters[i].rangingType == None || !playingParticles || !autoLoadNewParams;

    thisChanged = false;
    if (kernelNew.parameters[i].IsDifferentFrom(&(KERNEL.parameters[i]))) { anyChanged = true; thisChanged = true; }
    //if (thisChanged) paramNew.recountSteps(i); // TODO

    ImGui::Text(padString(KERNEL.parameters[i].name, maxNameLength).c_str());

    if (!changeAllowed)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor); // disabledText push
        PUSH_DISABLED_FRAME;
    }

    // Ranging
    if (playingParticles)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, disabledTextColor);
        PUSH_DISABLED_FRAME;
    }

    listAttrRanging(&(kernelNew.parameters[i]), kernelNew.parameters[i].rangingType != KERNEL.parameters[i].rangingType);

    dragFlag = (!playingParticles || kernelNew.parameters[i].rangingType == None) ? 0 : ImGuiSliderFlags_ReadOnly;

    switch (kernelNew.parameters[i].rangingType)
    {
    case Step:
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].min), "", "Min: ", kernelNew.parameters[i].min != KERNEL.parameters[i].min);
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].max), "MAX", "Max: ", kernelNew.parameters[i].max != KERNEL.parameters[i].max);
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].step), "STEP", "Step: ", kernelNew.parameters[i].step != KERNEL.parameters[i].step);
        kernelNew.parameters[i].CalcStepCount();
        ImGui::SameLine(); ImGui::Text((std::to_string(kernelNew.parameters[i].stepCount) + " steps").c_str());
        break;
    case Linear:
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].min), "", "Min: ", kernelNew.parameters[i].min != KERNEL.parameters[i].min);
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].max), "MAX", "Max: ", kernelNew.parameters[i].max != KERNEL.parameters[i].max);
        listAttrInt(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].stepCount), "STEPCOUNT", "Count: ", kernelNew.parameters[i].stepCount != KERNEL.parameters[i].stepCount);
        kernelNew.parameters[i].CalcStep();
        ImGui::SameLine(); ImGui::Text(("Step: " + (std::to_string(kernelNew.parameters[i].step))).c_str());
        break;
    case UniformRandom:
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].mean), "MEAN", "Mean: ", kernelNew.parameters[i].mean != KERNEL.parameters[i].mean);
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].deviation), "DEV", "Dev: ", kernelNew.parameters[i].deviation != KERNEL.parameters[i].deviation);
        listAttrInt(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].stepCount), "STEPCOUNT", "Count: ", kernelNew.parameters[i].stepCount != KERNEL.parameters[i].stepCount);
        break;
    case NormalRandom:
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].mean), "MU", "Mu: ", kernelNew.parameters[i].mean != KERNEL.parameters[i].mean);
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].deviation), "SIGMA", "Sigma: ", kernelNew.parameters[i].deviation != KERNEL.parameters[i].deviation);
        listAttrInt(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].stepCount), "STEPCOUNT", "Count: ", kernelNew.parameters[i].stepCount != KERNEL.parameters[i].stepCount);
        break;
    }

    if (playingParticles)
    {
        ImGui::PopStyleColor();
        POP_FRAME(3);
    }

    if (kernelNew.parameters[i].rangingType == None)
    {
        listAttrNumb(&(kernelNew.parameters[i]), &(kernelNew.parameters[i].min), "", "", kernelNew.parameters[i].min != KERNEL.parameters[i].min);
    }

    if (!changeAllowed) POP_FRAME(4); // disabledText popped as well
}

void heatmapRangingSelection(PlotWindow* window, ImPlotPlot* plot, HeatmapSizing* sizing)
{
    int stepX1 = 0;
    int stepY1 = 0;
    int stepX2 = 0;
    int stepY2 = 0;

    if (window->hmp.showActualDiapasons)
    {
        // Values
        stepX1 = stepFromValue(sizing->minX, sizing->stepX, plot->shiftSelect1Location.x);
        stepY1 = stepFromValue(sizing->minY, sizing->stepY, plot->shiftSelect1Location.y);
        stepX2 = stepFromValue(sizing->minX, sizing->stepX, plot->shiftSelect2Location.x);
        stepY2 = stepFromValue(sizing->minY, sizing->stepY, plot->shiftSelect2Location.y);
    }
    else
    {
        // Steps
        stepX1 = (int)floor(plot->shiftSelect1Location.x);
        stepY1 = (int)floor(plot->shiftSelect1Location.y);
        stepX2 = (int)floor(plot->shiftSelect2Location.x);
        stepY2 = (int)floor(plot->shiftSelect2Location.y);
    }

    //enabledParticles = false;
    playingParticles = false;

    int xMaxStep = sizing->hmp->typeX == PARAMETER ? KERNEL.parameters[sizing->hmp->indexX].TrueStepCount() :
        (sizing->hmp->typeX == VARIABLE ? KERNEL.variables[sizing->hmp->indexX].TrueStepCount() : 0);
    int yMaxStep = sizing->hmp->typeY == PARAMETER ? KERNEL.parameters[sizing->hmp->indexY].TrueStepCount() :
        (sizing->hmp->typeY == VARIABLE ? KERNEL.variables[sizing->hmp->indexY].TrueStepCount() : 0);

    if (sizing->hmp->typeX == VARIABLE)
    {
        kernelNew.variables[sizing->hmp->indexX].min = calculateValue(KERNEL.variables[sizing->hmp->indexX].min, KERNEL.variables[sizing->hmp->indexX].step, stepX1);
        kernelNew.variables[sizing->hmp->indexX].max = calculateValue(KERNEL.variables[sizing->hmp->indexX].min, KERNEL.variables[sizing->hmp->indexX].step, stepX2);
        kernelNew.variables[sizing->hmp->indexX].rangingType = Linear;
    }
    else
    {
        kernelNew.parameters[sizing->hmp->indexX].min = calculateValue(KERNEL.parameters[sizing->hmp->indexX].min, KERNEL.parameters[sizing->hmp->indexX].step, stepX1);
        kernelNew.parameters[sizing->hmp->indexX].max = calculateValue(KERNEL.parameters[sizing->hmp->indexX].min, KERNEL.parameters[sizing->hmp->indexX].step, stepX2);
        kernelNew.parameters[sizing->hmp->indexX].rangingType = Linear;
    }

    if (sizing->hmp->typeY == VARIABLE)
    {
        kernelNew.variables[sizing->hmp->indexY].min = calculateValue(KERNEL.variables[sizing->hmp->indexY].min, KERNEL.variables[sizing->hmp->indexY].step, stepY1);
        kernelNew.variables[sizing->hmp->indexY].max = calculateValue(KERNEL.variables[sizing->hmp->indexY].min, KERNEL.variables[sizing->hmp->indexY].step, stepY2);
        kernelNew.variables[sizing->hmp->indexY].rangingType = Linear;
    }
    else
    {
        kernelNew.parameters[sizing->hmp->indexY].min = calculateValue(KERNEL.parameters[sizing->hmp->indexY].min, KERNEL.parameters[sizing->hmp->indexY].step, stepY1);
        kernelNew.parameters[sizing->hmp->indexY].max = calculateValue(KERNEL.parameters[sizing->hmp->indexY].min, KERNEL.parameters[sizing->hmp->indexY].step, stepY2);
        kernelNew.parameters[sizing->hmp->indexY].rangingType = Linear;
    }

    autoLoadNewParams = false; // Otherwise the map immediately starts drawing the cut region

    if (window->hmp.isHeatmapAutoComputeOn) computeAfterShiftSelect = true;
}