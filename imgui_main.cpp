#include "imgui_main.hpp"

#include "gui/plotWindowMenu.h"
#include "gui/img_loading.h"
#include "gui/map_img.h"
#include "gui/fullscreen_funcs.h"
#include "gui/window_configs.h"

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

std::vector<PlotWindow> plotWindows;
int uniqueIds = 0; // Unique window IDs

bool spoilerVars = true;
bool spoilerStep = true;
bool spoilerParams = true;
bool ContinuationRed = false;

ApplicationSettings applicationSettings;

int playedBufferIndex = 0; // Buffer currently shown
int bufferToFillIndex = 0; // Buffer to send computations to
std::vector<int> attributeValueIndices; // Currently selected indices of ranging attributes
std::vector<int> attributeValueIndicesHires; // Currently selected indices of ranging attributes

bool autoLoadNewParams = false;

numb* dataBuffer = nullptr; // One variation local buffer
numb* particleBuffer = nullptr; // One step local buffer

// To use with fullscreen functionality
PlotWindow mainWindow(-1), graphBuilderWindow(-2), mapSettingsWindow(-3);
ImVec2 fullscreenSize;

float axisBuffer[18]{}; // 3 axis, 2 points
float rulerBuffer[153]{}; // 1 axis, 5 * 10 + 1 points

int computedSteps = 0; // Step count for the current computation
bool autofitAfterComputing = false; // Temporary flag to autofit computed data
bool autofitTimeSeries = false;
int currentTotalVariations = 0; // Current amount of variations, so we can compare and safely hot-swap the parameter values
bool executedOnLaunch = false; // Temporary flag to execute computations on launch if needed

bool enabledParticles = false; // Particles mode
bool playingParticles = false; // Playing animation
float particleSpeed = 5000.0f; // Steps per second
float particlePhase = 0.0f; // Animation frame cooldown
int particleStep = 0; // Current step of the computations to show
bool continuousComputingEnabled = true; // Continuously compute next batch of steps via double buffering
int bufferNo = 0;

bool lastHiresHasInfo = false;
bool lastHiresStopped = true;
std::chrono::steady_clock::time_point lastHiresStart;
std::chrono::steady_clock::time_point lastHiresEnd;

PlotWindow* colorsLUTfrom = nullptr;
int paintLUTsize = 32;

bool selectParticleTab = false;
bool selectOrbitTab = true;

bool computeAfterShiftSelect = false;
bool hiresComputeAfterShiftSelect = false;
bool autofitHeatmap;

bool OrbitRedraw = false; bool indSeriesReset = false;

// Temporary variables
uint64_t variation = 0;
uint64_t prevVariation = 0;
uint64_t variationHires = 0;
uint64_t prevVariationHires = 0;
int stride = 1;
float frameTime; // In seconds
float timeElapsed = 0.0f; // Total time elapsed, in seconds
int maxNameLength;
bool anyChanged;
bool popStyle;
ImGuiSliderFlags dragFlag;

bool rangingWindowEnabled = true;
bool graphBuilderWindowEnabled = true;

void deleteBuffers(bool deleteHires)
{
	if (!deleteHires)
	{
		computations[0].Clear();
		computations[1].Clear();
	}
	else
		computationHires.Clear();
	playedBufferIndex = 0;
	bufferToFillIndex = 0;

	for (int i = 0; i < plotWindows.size(); i++)
	{
		plotWindows[i].decay.DeleteBuffers();
	}
}

void resetTempBuffers(Computation* data)
{
	if (dataBuffer) delete[] dataBuffer;
	dataBuffer = new numb[(CUDA_kernel.steps + 1) * KERNEL.VAR_COUNT];

	if (particleBuffer) delete[] particleBuffer;
	particleBuffer = new numb[CUDA_marshal.totalVariations * KERNEL.VAR_COUNT];

	if (colorsLUTfrom != nullptr) colorsLUTfrom->hmp.paintLUT.Clear();
}

// Initialize the Attribute Value Indeces vector for ranging
void initAVI(bool hires)
{
	if (!hires)
	{
		attributeValueIndices.clear();
		for (int i = 0; i < kernelNew.VAR_COUNT + kernelNew.PARAM_COUNT; i++) attributeValueIndices.push_back(0);
	}
	else
	{
		attributeValueIndicesHires.clear();
		for (int i = 0; i < kernelHiresNew.VAR_COUNT + kernelHiresNew.PARAM_COUNT; i++) attributeValueIndicesHires.push_back(0);
	}
}

// Normal computing

void computing();

int asyncComputation()
{
	computations[bufferToFillIndex].ready = false;

	bool isFirstBatch = computations[1 - bufferToFillIndex].marshal.trajectory == nullptr; // Is another buffer null, only true when computing for the first time
	computations[bufferToFillIndex].isFirst = isFirstBatch;
	computations[bufferToFillIndex].calculateDeltaDecay = applicationSettings.calculateDeltaDecay;
	computations[bufferToFillIndex].threadsPerBlock = applicationSettings.threadsPerBlock;
	computations[bufferToFillIndex].marshal.kernel.CopyFrom(&KERNEL);

	int computationResult = compute(&(computations[bufferToFillIndex]));
	computedSteps = KERNEL.steps;

	if (isFirstBatch)
	{
		if (hiresIndex == IND_NONE) autofitAfterComputing = true;
		resetTempBuffers(&(computations[bufferToFillIndex]));
		initAVI(false);
	}

	computations[bufferToFillIndex].ready = true;
	for (int i = 0; i < plotWindows.size(); i++)
	{
		plotWindows[i].hmp.initClickedLocation = true;
		plotWindows[i].hmp.areValuesDirty = true;
		plotWindows[i].hireshmp.areValuesDirty = true;
	}

	if (continuousComputingEnabled) bufferToFillIndex = 1 - bufferToFillIndex;

	if (continuousComputingEnabled && bufferToFillIndex != playedBufferIndex) computing();

	return computationResult;
}

void computing()
{
	computations[bufferToFillIndex].bufferNo += 2; // One computation buffer only gets even trajectories, the other one gets odd trajectories
	computations[bufferToFillIndex].future = std::async(asyncComputation);
	//asyncComputation();
}

// Hi-res computing

void hiresComputing();

int hiresAsyncComputation()
{
	kernelHiresComputed.CopyFrom(&kernelHiresNew);
	kernelHiresComputed.PrepareAttributes();

	hiresComputationSetup();
	computationHires.marshal.kernel.CopyFrom(&kernelHiresComputed);
	computationHires.threadsPerBlock = applicationSettings.threadsPerBlock;

	lastHiresStart = std::chrono::steady_clock::now();
	lastHiresHasInfo = true;
	lastHiresStopped = false;
	int computationResult = compute(&computationHires);
	computationHires.ready = true;
	lastHiresEnd = std::chrono::steady_clock::now();
	lastHiresStopped = true;

	//resetTempBuffers(&computationHires);

	autofitAfterComputing = true;
	initAVI(true);

	for (int i = 0; i < plotWindows.size(); i++)
	{
		if (!plotWindows[i].isTheHiresWindow(hiresIndex)) continue;

		plotWindows[i].hmp.initClickedLocation = true;
		plotWindows[i].hmp.areValuesDirty = true;
		plotWindows[i].hireshmp.areValuesDirty = true;
	}

	return computationResult;
}

void hiresComputing()
{
	computationHires.future = std::async(hiresAsyncComputation);
	//hiresAsyncComputation();
}

void terminateUIBuffers()
{
	deleteBuffers(false);
	if (dataBuffer != nullptr)      { delete[] dataBuffer;      dataBuffer = nullptr; }
	if (particleBuffer != nullptr)  { delete[] particleBuffer;  particleBuffer = nullptr; }

	executedOnLaunch = false;
	playedBufferIndex = bufferToFillIndex = 0;
}

void unloadPlotWindows()
{
	for (PlotWindow w : plotWindows)
	{
		w.hmp.paintLUT.Clear();
		if (w.hmp.pixelBuffer != nullptr) delete[] w.hmp.pixelBuffer;
		// TODO: Delete hi-res, delete other buffers
	}
}

void initializeKernel(bool needTerminate) // needTerminate is obsolete
{
	terminateUIBuffers();

	unloadPlotWindows();
	plotWindows.clear();
	uniqueIds = 0;
	colorsLUTfrom = nullptr;

	kernelNew.CopyFrom(&KERNEL);
	kernelHiresNew.CopyFrom(&KERNEL);
	kernelHiresComputed.CopyFrom(&KERNEL);
	kernelNew.mapWeight = kernelHiresNew.mapWeight = kernelHiresComputed.mapWeight = 1.0f;

	initAVI(false);
	initAVI(true);

	computedSteps = 0;
	particleStep = 0;
	computeAfterShiftSelect = false;
}

void computationStatus(bool comp0, bool comp1)
{
	if (comp0)          ImGui::Text("Computing buffer 0...");
	else if (comp1)		ImGui::Text("Computing buffer 1...");
}

void switchPlayedBuffer()
{
	if (computations[1 - playedBufferIndex].ready)
	{
		playedBufferIndex = 1 - playedBufferIndex;
		autofitTimeSeries = true;
		particleStep = 0;
		bufferNo++;
		computing();
	}
}

void removeHeatmapLimits()
{
	for (int i = 0; i < plotWindows.size(); i++)
	{
		plotWindows[i].hmp.values.areHeatmapLimitsDefined = false;
		for (int ch = 0; ch < 3; ch++) plotWindows[i].hmp.channel[ch].areHeatmapLimitsDefined = false;
	}
}

void prepareAndCompute(bool hires)
{
	bool comp0inProgress = !computations[0].ready && computations[0].marshal.trajectory != nullptr;
	bool comp1inProgress = !computations[1].ready && computations[1].marshal.trajectory != nullptr;
	bool compHiresinProgress = !computationHires.ready && computationHires.marshal.variableInits != nullptr;
	if (comp0inProgress || comp1inProgress || compHiresinProgress)
	{
		printf("Preventing computing too fast!\n");
		return;
	}
	
	if (hires) computationHires.Clear();

	computations[0].isGPU = computations[1].isGPU = !applicationSettings.CPU_mode_interactive;
	computationHires.isGPU = !applicationSettings.CPU_mode_hires;
	executedOnLaunch = true;
	computeAfterShiftSelect = false;
	bufferNo = 0;
	particleStep = 0;
	deleteBuffers(hires);
	removeHeatmapLimits();

	// Time to steps

	if (!hires)
	{
		computations[0].bufferNo = -2;
		computations[1].bufferNo = -1;

		KERNEL.CopyFrom(&kernelNew);
		KERNEL.PrepareAttributes();

		computing();
	}
	else
	{
		hiresComputing();
	}

}

void releaseHeatmap(PlotWindow* window, bool isHires)
{
	if (!isHires)
	{
		if (window->hmp.texture != nullptr)
		{
			((ID3D11ShaderResourceView*)window->hmp.texture)->Release();
			window->hmp.texture = nullptr;
		}
	}
	else
	{
		if (window->hireshmp.texture != nullptr)
		{
			((ID3D11ShaderResourceView*)window->hireshmp.texture)->Release();
			window->hireshmp.texture = nullptr;
		}
	}
}

// Main code
int imgui_main(int, char**)
{
	ImGui_ImplWin32_EnableDpiAwareness();
	WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), LoadIcon(wc.hInstance, MAKEINTRESOURCE(IDI_ICON1)), nullptr, nullptr, nullptr, L"CUDAynamics", LoadIcon(wc.hInstance, MAKEINTRESOURCE(IDI_ICON1)) };
	::RegisterClassExW(&wc);
	guiHwnd = ::CreateWindowW(wc.lpszClassName, L"CUDAynamics", WS_OVERLAPPEDWINDOW, 100, 100, 400, 100, nullptr, nullptr, wc.hInstance, nullptr);

	if (!CreateDeviceD3D(guiHwnd))
	{
		CleanupDeviceD3D();
		::UnregisterClassW(wc.lpszClassName, wc.hInstance);
		return 1;
	}

	::ShowWindow(guiHwnd, SW_SHOWDEFAULT);
	::UpdateWindow(guiHwnd);

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
	io.ConfigViewportsNoTaskBarIcon = true; // TODO: Adapt to having it 'false' and creating a taskbar icon per every window for better traversing

	ImGui::StyleColorsDark();
	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	ImGui_ImplWin32_Init(guiHwnd);
	ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

	//io.Fonts->AddFontFromFileTTF("UbuntuMono-R.ttf", 24.0f);
	FontLoading(io);

	// Main loop
	bool work = true;

	char* plotNameBuffer = new char[64]();
	strcpy_s(plotNameBuffer, 5, "Plot");

	PlotType plotType = VarSeries;
	int selectedPlotVars[3]; selectedPlotVars[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVars[i] = -1;
	int selectedPlotVarsOrbitVer[3]; selectedPlotVarsOrbitVer[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVarsOrbitVer[i] = -1;
	std::set<int> selectedPlotVarsSet;
	std::set<int> selectedPlotMapsSetMetric;
	std::set<int> selectedPlotMapSetIndSeries;
	int selectedPlotMap = 0;
	int selectedPlotMCMaps[3]{ 0 };
	int selectedPlotMapMetric = 0;
	std::set<int> selectedPlotMapDecay;

#define RESET_GRAPH_BUILDER_SETTINGS \
	selectedPlotVars[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVars[i] = -1;  \
	selectedPlotVarsOrbitVer[0] = 0; for (int i = 1; i < 3; i++) selectedPlotVarsOrbitVer[i] = -1;  \
	selectedPlotVarsSet.clear();  \
	selectedPlotMapsSetMetric.clear();  \
	selectedPlotMapSetIndSeries.clear();  \
	selectedPlotMap = 0;  \
	for (int i = 0; i < 3; i++) selectedPlotMCMaps[i] = 0; \
	selectedPlotMapMetric = 0; \
	selectedPlotMapDecay.clear();

	prepareKernel();
	initializeKernel(false);

	try { loadWindows(); }
	catch (std::exception e) { printf(e.what()); }

	applicationSettings.globalFontSettings = &GlobalFontSettings;
	applicationSettings.Load();

	setVaryingAttributesToHeatmaps(plotWindows, KERNEL);

	fullscreenSize = ImVec2((float)GetSystemMetrics(SM_CXSCREEN), (float)GetSystemMetrics(SM_CYSCREEN));

	// Custom colormaps for multichannel colormaps
	ImU32 customColormapColors[2] = { 0xFF000000, 0xFF0000FF };
	ImPlotColormap multichannelHeatmapColormaps[3];
	multichannelHeatmapColormaps[0] = ImPlot::AddColormap("Rchannel", customColormapColors, 2, false);
	customColormapColors[1] = 0xFF00FF00;
	multichannelHeatmapColormaps[1] = ImPlot::AddColormap("Gchannel", customColormapColors, 2, false);
	customColormapColors[1] = 0xFFFF0000;
	multichannelHeatmapColormaps[2] = ImPlot::AddColormap("Bchannel", customColormapColors, 2, false);

	while (work)
	{
		if (fontNotDefault)
		{
			ImFont* defaultFont = GetFont(GlobalFontSettings.family, GlobalFontSettings.size, GlobalFontSettings.isBold, GlobalFontSettings.isItalic);
			if (defaultFont != nullptr) io.FontDefault = defaultFont;
		}

		IMGUI_WORK_BEGIN;
		SetupImGuiStyle(applicationSettings.appStyle, applicationSettings.cudaColor, applicationSettings.hiresColor, applicationSettings.openmpColor, 
			HIRES_ON, !HIRES_ON ? applicationSettings.CPU_mode_interactive : applicationSettings.CPU_mode_hires);

		if (fontNotDefault) ImGui::PushFont(GetFont(GlobalFontSettings.family, GlobalFontSettings.size, GlobalFontSettings.isBold, GlobalFontSettings.isItalic));

		timeElapsed += frameTime;
		float breath = (cosf(timeElapsed * 6.0f) + 1.0f) / 2.0f;
		float buttonBreathMult = 0.6f + breath * 0.3f;
		bool noComputedData = computations[0].marshal.trajectory == nullptr;

		if (particleStep > computedSteps) particleStep = computedSteps;

		// MAIN WINDOW

		style.WindowMenuButtonPosition = ImGuiDir_Left;
		ImGui::Begin("CUDAynamics", &work, ImGuiWindowFlags_MenuBar);

		mainWindowMenu();

		// Selecting kernel
		if (computations[0].IsInProgress() || computations[1].IsInProgress() || computationHires.IsInProgress()) ImGui::BeginDisabled();
		if (ImGui::BeginCombo("##selectingKernel", KERNEL.name.c_str()))
		{
			for (auto k : kernels)
			{
				bool isSelected = k.first == selectedKernel;
				ImGuiSelectableFlags selectableFlags = 0;
				if (ImGui::Selectable(k.second.name.c_str(), isSelected, selectableFlags))
				{
					saveWindows();
					RESET_GRAPH_BUILDER_SETTINGS;

					selectedKernel = k.first;
					prepareKernel();
					initializeKernel(true);
					playingParticles = false;

					loadWindows();
					setVaryingAttributesToHeatmaps(plotWindows, KERNEL);
				}
			}

			ImGui::EndCombo();
		}
		if (computations[0].IsInProgress() || computations[1].IsInProgress() || computationHires.IsInProgress()) ImGui::EndDisabled();

		// Parameters & Variables

		maxNameLength = 0;
		for (int i = 0; i < KERNEL.PARAM_COUNT; i++) if (KERNEL.parameters[i].name.length() > maxNameLength) maxNameLength = (int)KERNEL.parameters[i].name.length();
		for (int i = 0; i < KERNEL.VAR_COUNT; i++) if (KERNEL.variables[i].name.length() > maxNameLength) maxNameLength = (int)KERNEL.variables[i].name.length();

		anyChanged = false;
		popStyle = false;

		if (spoilerVars) ImGui::SetNextItemOpen(true);
		if (ImGui::TreeNode("Variables##VariablesList"))
		{
			if (ImGui::BeginTable("##VarTable", 6))
			{
				ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, GlobalFontSettings.size * 4.0f);
				ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, GlobalFontSettings.size * 5.0f);
				for (int c = 2; c < 6; c++) ImGui::TableSetupColumn(nullptr);
				for (int i = 0; i < KERNEL.VAR_COUNT - (KERNEL.stepType == ST_Variable ? 1 : 0); i++) listVariable(i);
				ImGui::EndTable();
			}
			ImGui::TreePop();
			spoilerVars = true;
		}
		else
			spoilerVars = false;
		
		if (KERNEL.stepType != ST_Discrete)
		{
			if (spoilerStep) ImGui::SetNextItemOpen(true);
			if (ImGui::TreeNode("Step##StepList"))
			{
				if (ImGui::BeginTable("##StepTable", 6))
				{
					ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, GlobalFontSettings.size * 4.0f);
					ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, GlobalFontSettings.size * 5.0f);
					for (int c = 2; c < 6; c++) ImGui::TableSetupColumn(nullptr);
					if (KERNEL.stepType == ST_Variable) listVariable(KERNEL.VAR_COUNT - 1);
					if (KERNEL.stepType == ST_Parameter) listParameter(KERNEL.PARAM_COUNT - 1);
					ImGui::EndTable();
				}

				ImGui::TreePop();
				spoilerStep = true;
			}
			else
				spoilerStep = false;
		}

		bool applicationProhibited = false;
		if (spoilerParams) ImGui::SetNextItemOpen(true);
		if (ImGui::TreeNode("Parameters##ParametersList"))
		{
			if (ImGui::BeginTable("##ParamTable", 6, ImGuiTableFlags_NoClip))
			{
				ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, GlobalFontSettings.size * 4.0f);
				ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, GlobalFontSettings.size * 5.0f);
				for (int c = 2; c < 6; c++) ImGui::TableSetupColumn(nullptr);
				for (int i = 0; i < KERNEL.PARAM_COUNT - (KERNEL.stepType == ST_Parameter ? 1 : 0); i++)
				{
					if (KERNEL.parameters[i].rangingType != RT_Enum)
						listParameter(i);
					else
						listEnum(i);
				}
				ImGui::EndTable();
			}

			// Parameter auto-loading
			bool tempAutoLoadNewParams = autoLoadNewParams;

			if (ImGui::Checkbox("Apply parameter changes automatically", &(tempAutoLoadNewParams)))
			{
				autoLoadNewParams = !autoLoadNewParams;
				if (autoLoadNewParams) kernelNew.CopyParameterValuesFrom(&KERNEL);
				else KERNEL.CopyParameterValuesFrom(&kernelNew);
			}
			TOOLTIP("Automatically applies new parameter values to the new buffers mid-playback");

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
						ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
						PUSH_DISABLED_FRAME;
					}
					if (ImGui::Button("Apply") && !applicationProhibited)
					{
						KERNEL.CopyParameterValuesFrom(&kernelNew);
					}
					if (applicationProhibited)
					{
						POP_FRAME(4);
					}
				}
			}
			ImGui::TreePop();
			spoilerParams = true;
		}
		else
			spoilerParams = false;

		// Simulation

		ImGui::SeparatorText("Simulation");

		// Hi-res info and disabling in the main window
		if (HIRES_ON)
		{
			std::string indexName = indices[hiresIndex].name;
			ImGui::AlignTextToFramePadding();
			ImGui::Text(("Hi-Res Mode enabled for " + indexName + " ").c_str());
			ImGui::SameLine();
			if (ImGui::Button("Disable Hi-Res Mode")) hiresIndex = IND_NONE;
		}

		if (!HIRES_ON)
		{
			unsigned long long tempTotalVariations = 1;
			for (int v = 0; v < KERNEL.VAR_COUNT; v++)      if (KERNELNEWCURRENT.variables[v].TrueStepCount() > 1)     tempTotalVariations *= KERNELNEWCURRENT.variables[v].stepCount;
			for (int p = 0; p < KERNEL.PARAM_COUNT; p++)    if (KERNELNEWCURRENT.parameters[p].TrueStepCount() > 1)    tempTotalVariations *= KERNELNEWCURRENT.parameters[p].stepCount;
			unsigned long long singleBufferNumbSize = ((tempTotalVariations * KERNEL.VAR_COUNT) * (KERNELNEWCURRENT.steps + 1)) * sizeof(numb);
			ImGui::Text(("Buffer memory: " + memoryString(singleBufferNumbSize) + " (" + std::to_string(singleBufferNumbSize) + " bytes)").c_str());
		}

		frameTime = 1.0f / io.Framerate; ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

		ImGui::PushItemWidth(200.0f);

		// Hi-res variations per computation
		if (HIRES_ON)
		{
			int vpp = applicationSettings.varPerParallelization;
			ImGui::InputInt("Variations per computation", &vpp, 1, 10, 0);
			applicationSettings.varPerParallelization = vpp;
			TOOLTIP("Variations per each division of the heatmap");
		}

		// Steps
		if (playingParticles) PUSH_DISABLED_FRAME
		popStyle = false;

		float stepSize = KERNELNEWCURRENT.GetStepSize();

		if ((!KERNELNEWCURRENT.usingTime && KERNELNEWCURRENT.steps != KERNELSAVEDCURRENT.steps) || (KERNELNEWCURRENT.usingTime && KERNELNEWCURRENT.time != KERNELSAVEDCURRENT.time))
		{
			anyChanged = true;
			PUSH_UNSAVED_FRAME;
			popStyle = true;
		}
		if (!KERNELNEWCURRENT.usingTime)
		{
			ImGui::InputInt("##Steps", &(KERNELNEWCURRENT.steps), 1, 1000, playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
			TOOLTIP("Amount of computed steps, the trajectory will be (1 + 'steps') steps long, including the initial state");
			KERNELNEWCURRENT.time = KERNELNEWCURRENT.steps * stepSize;
		}
		else
		{
			ImGui::InputFloat("##Time(s)", &(KERNELNEWCURRENT.time), 1.0f, 10.0f, "%.3f", playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
			TOOLTIP("Simulation time");
			KERNELNEWCURRENT.steps = (int)(KERNELNEWCURRENT.time / stepSize);
		}
		ImGui::SameLine();
		if (ImGui::BeginCombo("##stepsOrTime", !KERNELNEWCURRENT.usingTime ? "Steps" : "Time"))
		{
			if (ImGui::Selectable("Steps", !KERNELNEWCURRENT.usingTime)) KERNELNEWCURRENT.usingTime = false;
			if (ImGui::Selectable("Time", KERNELNEWCURRENT.usingTime)) KERNELNEWCURRENT.usingTime = true;

			ImGui::EndCombo();
		}

		if (popStyle) POP_FRAME(3);
		if (playingParticles)
		{
			POP_FRAME(3);
		}

		// Hi-res buffers per variation
		if (HIRES_ON)
		{
			ImGui::InputInt("Buffers", &(computationHires.buffersPerVariation), 1, 10, 0);
			TOOLTIP("Steps * Buffers = total steps in hi-res computation");
		}

		// Transient steps
		ImGui::PushItemWidth(200.0f);
		if (playingParticles)
		{
			PUSH_DISABLED_FRAME;
		}
		popStyle = false;

		if ((!KERNELNEWCURRENT.usingTime && KERNELNEWCURRENT.transientSteps != KERNELSAVEDCURRENT.transientSteps) || (KERNELNEWCURRENT.usingTime && KERNELNEWCURRENT.transientTime != KERNELSAVEDCURRENT.transientTime))
		{
			anyChanged = true;
			PUSH_UNSAVED_FRAME;
			popStyle = true;
		}
		if (!KERNELNEWCURRENT.usingTime)
		{
			ImGui::InputInt("Transient steps##Transient steps", &(KERNELNEWCURRENT.transientSteps), 1, 1000, playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
			TOOLTIP("Steps to skip, including the initial state");
			KERNELNEWCURRENT.transientTime = KERNELNEWCURRENT.transientSteps * stepSize;
		}
		else
		{
			ImGui::InputFloat("Transient time##Transient time(s)", &(KERNELNEWCURRENT.transientTime), 1.0f, 10.0f, "%.3f", playingParticles ? ImGuiInputTextFlags_ReadOnly : 0);
			TOOLTIP("Time to skip");
			KERNELNEWCURRENT.transientSteps = (int)(KERNELNEWCURRENT.transientTime / stepSize);
		}

		if (popStyle) POP_FRAME(3);
		if (playingParticles)
		{
			POP_FRAME(3);
		}

		variation = 0;

		ImGui::NewLine();
		if (!HIRES_ON)
		{
			if (ImGui::RadioButton("Orbits", !enabledParticles))    enabledParticles = false;
			ImGui::SameLine();
			if (ImGui::RadioButton("Particles", enabledParticles))  enabledParticles = true;

			// PARTICLES MODE
			ImGui::SetNextItemWidth(200.0f);
			ImGui::DragFloat("Animation speed, steps/s", &(particleSpeed), 1.0f);
			TOOLTIP("Playback speed of the evolution in Particles mode");
			if (particleSpeed < 0.0f) particleSpeed = 0.0f;

			ImGui::SameLine();
			if (computations[playedBufferIndex].timeElapsed > 0.0f)
			{
				float buffersPerSecond = 1000.0f / computations[playedBufferIndex].timeElapsed;
				int stepsPerSecond = (int)(computedSteps * buffersPerSecond);
				ImGui::Text(("(max " + std::to_string(stepsPerSecond) + " before stalling)").c_str());
			}
			else
			{
				ImGui::Text("(pending)");
			}
			TOOLTIP("Predicted speed that allows for seamless playback");

			ImGui::SetNextItemWidth(200.0f);
			ImGui::DragInt("##Animation step", &(particleStep), 1.0f, 0, KERNEL.steps);
			ImGui::SameLine();
			ImGui::Text(("Animation step" + (continuousComputingEnabled ? " (total step " + std::to_string(bufferNo * KERNEL.steps + particleStep) + ")" : "")).c_str());

			if (ImGui::Button("Reset to step 0")) particleStep = 0;

			if (anyChanged)
			{
				ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
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

			ImGui::NewLine();

			if (ImGui::Button("Next buffer"))
			{
				switchPlayedBuffer(); OrbitRedraw = true;
			}
		}

		// Ranging

		Kernel* rangingKernel = !HIRES_ON ? &(KERNEL) : &kernelHiresComputed;
		Kernel* rangingKernelNew = !HIRES_ON ? &kernelNew : &kernelHiresNew;
		Computation* rangingCmp = !HIRES_ON ? &(computations[playedBufferIndex]) : &computationHires;
		std::vector<int>* rangingAVI = !HIRES_ON ? &attributeValueIndices : &attributeValueIndicesHires;
		if (rangingCmp->ready)
		{
			for (int i = 0; i < KERNEL.VAR_COUNT + KERNEL.PARAM_COUNT; i++)
			{
				bool isVar = i < KERNEL.VAR_COUNT;
				Attribute* attr = isVar ? &(rangingCmp->marshal.kernel.variables[i]) : &(rangingCmp->marshal.kernel.parameters[i - rangingKernel->VAR_COUNT]);
				Attribute* kernelNewAttr = isVar ? &(rangingKernelNew->variables[i]) : &(rangingKernelNew->parameters[i - rangingKernel->VAR_COUNT]);
				bool isEnum = attr->rangingType == RT_Enum;
				if (attr->TrueStepCount() == 1) continue;

				std::string name = attr->name; name.resize(maxNameLength);
				ImGui::Text(name.c_str()); ImGui::SameLine();
				int index = (*rangingAVI)[i];
				ImGui::SetNextItemWidth(150.0f);
				ImGui::SliderInt(("##RangingNo_" + std::to_string(i)).c_str(), &index, 0, attr->stepCount - 1, "Step: %d");
				(*rangingAVI)[i] = index;

				if (!isEnum)
				{
					ImGui::SameLine();
					ImGui::Text(("Value: " + std::to_string(calculateValue(attr->min, attr->step, index))).c_str());
				}
				else
				{
					ImGui::SameLine();

					int selEnum = index;
					for (int e = 0; e < MAX_ENUMS; e++)
					{
						if (attr->enumEnabled[e])
						{
							if (selEnum == 0)
								TEXT_AND_BREAK(attr->enumNames[e].c_str())
							else
								selEnum--;
						}
					}
				}

				ImGui::SameLine();
				if (ImGui::Button(("Fix##FixRanging" + std::to_string(i)).c_str()))
				{
					if (!isEnum)
					{
						kernelNewAttr->rangingType = RT_None;
						kernelNewAttr->min = calculateValue(attr->min, attr->step, index);
					}
					else
					{
						for (int i = 0; i < attr->enumCount; i++) kernelNewAttr->enumEnabled[i] = false;
						kernelNewAttr->enumEnabled[index] = true;
					}

					playingParticles = false;
				}
			}

			steps2Variation(!HIRES_ON ? &variation : &variationHires, &(rangingAVI->data()[0]), rangingKernel);
		}

		// COMMON
		// default button color is 0.137 0.271 0.427
		bool playBreath = noComputedData || (anyChanged && (!playingParticles || !enabledParticles));

		OrbitRedraw = playingParticles || computations[0].IsInProgress() || computations[1].IsInProgress() || computationHires.IsInProgress();

		if (!HIRES_ON)
		{
			ImVec4 buttonColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);
			if (playBreath) ImGui::PushStyleColor(ImGuiCol_Button, 
				ImVec4(buttonColor.x * buttonBreathMult, buttonColor.y * buttonBreathMult, buttonColor.z * buttonBreathMult, 1.0f));
			if (ImGui::Button("= COMPUTE =") || (KERNEL.executeOnLaunch && !executedOnLaunch) || computeAfterShiftSelect)
			{
				prepareAndCompute(false); OrbitRedraw = true; indSeriesReset = true;
			}
			if (playBreath) ImGui::PopStyleColor();
			if (!playingParticles) computationStatus(computations[0].IsInProgress(), computations[1].IsInProgress());
		}     
		else
		{
			if (computeAfterShiftSelect) // For shift-clicking the hires map
			{
				prepareAndCompute(false); OrbitRedraw = true; indSeriesReset = true;
			}

			// Hi-res compute button
			if (ImGui::Button("= HI-RES COMPUTE =") || hiresComputeAfterShiftSelect)
			{
				prepareAndCompute(true); OrbitRedraw = true; indSeriesReset = true;
			}

			if (ImGui::Button("Compute this variation"))
			{
				hiresShiftClickCompute();
			}
		}
		if (computationHires.IsInProgress())
		{
			float progressPercentage = (computationHires.variationsFinished * 100.0f) / computationHires.marshal.totalVariations;
			ImGui::Text((std::to_string(computationHires.variationsFinished) + "/" + std::to_string(computationHires.marshal.totalVariations) + " computed (" +
				std::to_string(progressPercentage) + "%%)").c_str());
		}
		if (HIRES_ON && lastHiresHasInfo)
		{
			ImGui::Text("Hi-Res time: %f s", (float)(std::chrono::duration_cast<std::chrono::milliseconds>
				((lastHiresStopped ? lastHiresEnd : std::chrono::steady_clock::now()) - lastHiresStart).count()) / 1000.0f);
		}

		if (anyChanged && !autoLoadNewParams)
		{
			if (ImGui::Button("Reset changed values"))
			{
				kernelNew.CopyFrom(&KERNEL);
			}
		}

		ImGui::End();
		
		// Graph Builder

		ImGui::Begin("Graph Builder", nullptr);

		// Type
		ImGui::Text("Plot type ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(250.0f);
		if (ImGui::BeginCombo("##Plot type", (plottypes[plotType]).c_str()))
		{
			for (int t = 0; t < PlotType_COUNT; t++)
			{
				bool isSelected = plotType == t;
				ImGuiSelectableFlags selectableFlags = 0;
				if (ImGui::Selectable(plottypes[t].c_str(), isSelected, selectableFlags))
				{
					plotType = (PlotType)t;
					RESET_GRAPH_BUILDER_SETTINGS;
				}
			}

			ImGui::EndCombo();
		}

		int indicesSize;

		switch (plotType)
		{
		case VarSeries:

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
					break;
				}
				ImGui::SameLine();
				ImGui::Text(("- " + KERNEL.variables[v].name).c_str());
			}

			break;

		case Phase:
		case Phase2D:
			ImGui::SetNextItemWidth(150.0f);
			for (int sv = 0; sv < (plotType == Phase ? 3 : 2); sv++)
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
			break;

		case Orbit:
			ImGui::SetNextItemWidth(150.0f);
			for (int sv = 0; sv < 1; sv++)
			{
				ImGui::Text("Variable ");
				ImGui::SameLine();
				if (ImGui::BeginCombo(("##Plot builder var " + std::to_string(sv + 1)).c_str(), selectedPlotVarsOrbitVer[sv] > -1 ? KERNEL.variables[selectedPlotVarsOrbitVer[sv]].name.c_str() : "-"))
				{
					for (int v = 0; v < KERNEL.VAR_COUNT; v++)
					{
						bool isSelected = selectedPlotVarsOrbitVer[sv] == v;
						ImGuiSelectableFlags selectableFlags = 0;
						if (ImGui::Selectable(v > -1 ? KERNEL.variables[v].name.c_str() : "-", isSelected, selectableFlags)) selectedPlotVarsOrbitVer[sv] = v;

					}
					ImGui::EndCombo();
				}
			}
			break;

		case Heatmap:
			if (selectedPlotMap >= indices.size()) selectedPlotMap = 0;

			ImGui::Text("Index");
			ImGui::SameLine();
			mapSelectionCombo("##Plot builder map index selection", selectedPlotMap, false);
			break;
		case MCHeatmap:
			break;

		case Decay:
			// Index adding combo

			ImGui::Text("Add index");
			ImGui::SameLine();
			if (ImGui::BeginCombo("##Add index combo", " ", ImGuiComboFlags_NoPreview))
			{
				for (int i = 0; i < (int)indices.size(); i++)
				{
					bool isSelected = selectedPlotMapDecay.find(i) != selectedPlotMapDecay.end();
					ImGuiSelectableFlags selectableFlags = 0;

					if (isSelected) selectableFlags = ImGuiSelectableFlags_Disabled;
					if (ImGui::Selectable(indices[(AnalysisIndex)i].name.c_str())) selectedPlotMapDecay.insert(i);
				}

				ImGui::EndCombo();
			}

			// Index list

			for (const int i : selectedPlotMapDecay)
			{
				if (ImGui::Button(("x##" + std::to_string(i)).c_str()))
				{
					selectedPlotMapDecay.erase(i);
					break;
				}
				ImGui::SameLine();
				ImGui::Text(("- " + indices[(AnalysisIndex)i].name).c_str());
			}

			break;

		case Metric:
			ImGui::Text("Add indices");
			ImGui::SameLine();

			indicesSize = (int)indices.size();
			if (ImGui::BeginCombo("##Add index combo", "", ImGuiComboFlags_NoPreview))
			{
				for (int i = 0; i < indicesSize; i++)
				{
					bool isSelected = selectedPlotMapsSetMetric.find(i) != selectedPlotMapsSetMetric.end();
					if (ImGui::Selectable(indices[(AnalysisIndex)i].name.c_str(), isSelected, isSelected ? ImGuiSelectableFlags_Disabled : 0)) selectedPlotMapsSetMetric.insert(i);
				}

				ImGui::EndCombo();
			}

			// List with removal buttons

			for (const int i : selectedPlotMapsSetMetric)
			{
				if (ImGui::Button(("x##" + std::to_string(i)).c_str()))
				{
					selectedPlotMapsSetMetric.erase(i);
					break;
				}
				ImGui::SameLine();
				ImGui::Text(("- " + indices[(AnalysisIndex)i].name).c_str());
			}
			break;
		case IndSeries:
		{
			ImGui::Text("Add indeces");
			ImGui::SameLine();

			indicesSize = (int)indices.size();
			if (ImGui::BeginCombo("##Add index combo", "", ImGuiComboFlags_NoPreview))
			{
				for (int i = 0; i < indicesSize; i++)
				{
					bool isSelected = selectedPlotMapSetIndSeries.find(i) != selectedPlotMapSetIndSeries.end();
					if (ImGui::Selectable(indices[(AnalysisIndex)i].name.c_str(), isSelected, isSelected ? ImGuiSelectableFlags_Disabled : 0)) selectedPlotMapSetIndSeries.insert(i);
				}

				ImGui::EndCombo();
			}

			// List with removal buttons

			for (const int i : selectedPlotMapSetIndSeries)
			{
				if (ImGui::Button(("x##" + std::to_string(i)).c_str()))
				{
					selectedPlotMapSetIndSeries.erase(i);
					break;
				}
				ImGui::SameLine();
				ImGui::Text(("- " + indices[(AnalysisIndex)i].name).c_str());
			}
		}
		break;
		}

		// Sanity checks
		bool noMistakes = true;
		switch (plotType)
		{
		case VarSeries:
			noMistakes = selectedPlotVarsSet.size() > 0;
			break;
		case IndSeries:
			noMistakes = selectedPlotMapSetIndSeries.size() > 0;
			break;
		case Decay:
			noMistakes = selectedPlotMapDecay.size() > 0;
			break;
		case Metric:
			noMistakes = selectedPlotMapsSetMetric.size() > 0;
			break;
		case Orbit:
			noMistakes = selectedPlotVarsOrbitVer[0] > -1;
			break;
		case Phase:
			noMistakes = selectedPlotVars[0] > -1 && selectedPlotVars[1] > -1 && selectedPlotVars[2] > -1;
			break;
		case Phase2D:
			noMistakes = selectedPlotVars[0] > -1 && selectedPlotVars[1] > -1;
			break;
		}

		if (!noMistakes)
		{
			ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
			PUSH_DISABLED_FRAME;
		}

		if (ImGui::Button("Create graph") && noMistakes)
		{
			PlotWindow plotWindow = PlotWindow(uniqueIds++, "Plot_", true);
			plotWindow.type = plotType;
			plotWindow.newWindow = true;

			if (plotType == VarSeries) plotWindow.AssignVariables(selectedPlotVarsSet);
			if (plotType == Phase) plotWindow.AssignVariables(selectedPlotVars);
			if (plotType == Phase2D) plotWindow.AssignVariables(selectedPlotVars);
			if (plotType == Heatmap) plotWindow.AssignVariables(selectedPlotMap);
			if (plotType == MCHeatmap) plotWindow.AssignVariables(selectedPlotMCMaps);
			if (plotType == Orbit) plotWindow.AssignVariables(selectedPlotVarsOrbitVer);
			if (plotType == Metric) plotWindow.AssignVariables(selectedPlotMapsSetMetric);
			if (plotType == IndSeries) { plotWindow.AssignVariables(selectedPlotMapSetIndSeries); plotWindow.firstBufferNo = (computations[playedBufferIndex]).bufferNo; plotWindow.prevbufferNo = (computations[playedBufferIndex]).bufferNo; }
			if (plotType == Decay) plotWindow.AssignVariables(selectedPlotMapDecay);

			int indexOfColorsLutFrom = -1;
			if (colorsLUTfrom != nullptr)
			{
				for (int i = 0; i < plotWindows.size() && indexOfColorsLutFrom == -1; i++)
				{
					if (&(plotWindows[i]) == colorsLUTfrom) indexOfColorsLutFrom = i;
				}
			}

			setVaryingAttributesToOneWindow(plotWindow, KERNEL);
			plotWindows.push_back(plotWindow);

			if (indexOfColorsLutFrom != -1) colorsLUTfrom = &(plotWindows[indexOfColorsLutFrom]);

			saveWindows();
		}

		if (!noMistakes) POP_FRAME(4);

		ImGui::End();

		// Map settings window

		ImGui::Begin("Analysis Settings", nullptr);

		Kernel* krn = HIRES_ON ? &kernelHiresNew : &kernelNew; // Workaround for Win11

		for (int anfunc = 0; anfunc < (int)AnalysisFunction::ANF_COUNT; anfunc++)
		{
			if (hiresIndex != IND_NONE && indices[hiresIndex].function == (AnalysisFunction)anfunc) AddBackgroundToElement(ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive), false);
			if (ImGui::TreeNode(std::string(AnFuncNames[anfunc] + std::string("##AnFunc") + std::to_string(anfunc)).c_str()))
			{
				switch ((AnalysisFunction)anfunc)
				{
				case AnalysisFunction::ANF_MINMAX:
					krn->analyses.MINMAX.DisplaySettings(krn->variables);
					break;
				case AnalysisFunction::ANF_LLE:
					krn->analyses.LLE.DisplaySettings(krn->variables);
					break;
				case AnalysisFunction::ANF_PERIOD:
					krn->analyses.PERIOD.DisplaySettings(krn->variables);
					break;
				case AnalysisFunction::ANF_PV:
					krn->analyses.PV.DisplaySettings(krn->variables);
					break;
				}

				std::vector<AnalysisIndex> anfuncIndices = anfunc2indices((AnalysisFunction)anfunc);
				ImGui::Text("Calculate indices:");
				for (AnalysisIndex index : anfuncIndices)
				{
					bool indexEnabled = indices[index].enabled;
					bool hiresEnabled = hiresIndex == index;
					if (hiresIndex != IND_NONE) ImGui::BeginDisabled();
					if (hiresIndex == IND_NONE)
						ImGui::Checkbox(("##IndexEnabled" + indices[index].name).c_str(), &indexEnabled);
					else
						ImGui::Checkbox(("##HiresIndexEnabled" + indices[index].name).c_str(), &hiresEnabled);
					indices[index].enabled = indexEnabled;

					ImGui::SameLine();
					ImGui::Text(indices[index].name.c_str());
					if (hiresIndex != IND_NONE) ImGui::EndDisabled();
				}

				ImGui::TreePop();
			}
		}

		ImGui::End();

		bool toAutofit = autofitAfterComputing;
		autofitAfterComputing = false;
		bool toAutofitTimeSeries = autofitTimeSeries;
		autofitTimeSeries = false;

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
			std::string windowName = "Plot " + std::to_string(window->id) + "##" + window->name + std::to_string(window->id);
			std::string plotName = windowName + "_plot";

			if (window->newWindow)
			{
				HWND g_hwnd = nullptr;
				HMONITOR hMonitor = MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTONEAREST);
				MONITORINFO monitorInfo = { sizeof(MONITORINFO) };
				GetMonitorInfo(hMonitor, &monitorInfo);

				float monitorWidth = (float)(monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left);
				float monitorHeight = (float)(monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top);
				ImVec2 monitorCenter = ImVec2(
					(monitorInfo.rcMonitor.left + monitorInfo.rcMonitor.right) * 0.5f,
					(monitorInfo.rcMonitor.top + monitorInfo.rcMonitor.bottom) * 0.5f
				);

				ImVec2 baseWindowSize = ImVec2(monitorWidth * 0.4f, monitorHeight * 0.5f);
				int offsetStep = window->id % 5;
				ImVec2 offset = ImVec2(30.0f * offsetStep, 30.0f * offsetStep);
				ImVec2 finalPos = ImVec2(monitorCenter.x + offset.x, monitorCenter.y + offset.y);

				ImGui::SetNextWindowPos(finalPos, 0, ImVec2(0.5f, 0.5f));
				ImGui::SetNextWindowSize(baseWindowSize, 0);

				window->newWindow = false;
			}

			FullscreenActLogic(window, &fullscreenSize);
			if (window->isFrozen) ImGui::PushStyleColor(ImGuiCol_Text, CUSTOM_COLOR(DisabledText));
			ImGui::Begin(windowName.c_str(), &(window->active), ImGuiWindowFlags_MenuBar);
			if (window->isFrozen) POP_FRAME(1);
			FullscreenButtonPressLogic(window, ImGui::GetCurrentWindow());

			if (fontNotDefault && window->overrideFontSettings)
				ImGui::PushFont(GetFont(window->localFontSettings.family, window->localFontSettings.size, window->localFontSettings.isBold, window->localFontSettings.isItalic));

			autofitHeatmap = false;

			// Menu
			plotWindowMenu(window);

			// Heatmap axes
			if (window->type == Heatmap || window->type == MCHeatmap)
			{
				AnalysisIndex mapIndex = (AnalysisIndex)window->variables[0];
				bool isHires = window->isTheHiresWindow(hiresIndex);
				HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;
				Kernel* krnl = isHires ? &kernelHiresComputed : &(KERNEL);
				Port* port;
				bool isSingleValue = true;
				if (window->type == Heatmap)
				{
					port = index2port(krnl->analyses, mapIndex);
				}

				int prevIndexX = heatmap->indexX;
				int prevIndexY = heatmap->indexY;
				int prevTypeX = heatmap->typeX;
				int prevTypeY = heatmap->typeY;
				int prevValueIndex = heatmap->values.mapValueIndex;

				bool showMapValueInput = window->type == Heatmap && !isSingleValue;
				int columns = showMapValueInput ? 4 : 3;

				if (ImGui::BeginTable((plotName + "_axisTable").c_str(), columns))
				{
					ImGui::TableSetupColumn(nullptr);
					ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, 40.0f);
					ImGui::TableSetupColumn(nullptr);
					if (columns == 4) ImGui::TableSetupColumn(nullptr);
					ImGui::TableNextRow();

					ImGui::TableSetColumnIndex(0);
					ImGui::SetNextItemWidth(-1);
					if (ImGui::BeginCombo(("##" + windowName + "_axisX").c_str(),
						heatmap->typeX == MDT_Variable ? krnl->variables[heatmap->indexX].name.c_str() : krnl->parameters[heatmap->indexX].name.c_str()))
					{
						for (int v = 0; v < krnl->VAR_COUNT; v++)
						{
							if (ImGui::Selectable(krnl->variables[v].name.c_str()))
							{
								heatmap->indexX = v;
								heatmap->typeX = MDT_Variable;
							}
						}

						for (int p = 0; p < krnl->PARAM_COUNT; p++)
						{
							if (ImGui::Selectable(krnl->parameters[p].name.c_str()))
							{
								heatmap->indexX = p;
								heatmap->typeX = MDT_Parameter;
							}
						}

						ImGui::EndCombo();
					}

					ImGui::TableSetColumnIndex(1);
					if (ImGui::Button(("<>##" + windowName + "_flipAxesButton").c_str()))
					{
						int tempIndex = heatmap->indexX;
						MapDimensionType tempMDT = heatmap->typeX;
						heatmap->indexX = heatmap->indexY;
						heatmap->typeX = heatmap->typeY;
						heatmap->indexY = tempIndex;
						heatmap->typeY = tempMDT;
					}

					ImGui::TableSetColumnIndex(2);
					ImGui::SetNextItemWidth(-1);
					if (ImGui::BeginCombo(("##" + windowName + "_axisY").c_str(),
						heatmap->typeY == MDT_Variable ? krnl->variables[heatmap->indexY].name.c_str() : krnl->parameters[heatmap->indexY].name.c_str()))
					{
						for (int v = 0; v < krnl->VAR_COUNT; v++)
						{
							if (ImGui::Selectable(krnl->variables[v].name.c_str()))
							{
								heatmap->indexY = v;
								heatmap->typeY = MDT_Variable;
							}
						}

						for (int p = 0; p < krnl->PARAM_COUNT; p++)
						{
							if (ImGui::Selectable(krnl->parameters[p].name.c_str()))
							{
								heatmap->indexY = p;
								heatmap->typeY = MDT_Parameter;
							}
						}

						ImGui::EndCombo();
					}

					if (showMapValueInput)
					{
						ImGui::TableSetColumnIndex(3);
						ImGui::SetNextItemWidth(-1);
						mapValueSelectionCombo((AnalysisIndex)mapIndex, -1, windowName, heatmap);
					}

					ImGui::EndTable();
				}

				if (prevIndexX != heatmap->indexX || prevIndexY != heatmap->indexY || prevTypeX != heatmap->typeX || prevTypeY != heatmap->typeY || prevValueIndex != heatmap->values.mapValueIndex)
				{
					heatmap->areValuesDirty = true;
					heatmap->values.areHeatmapLimitsDefined = false;
					for (int ch = 0; ch < 3; ch++) heatmap->channel[ch].areHeatmapLimitsDefined = false;
					autofitHeatmap = true;
				}
			}

			if (window->type == Orbit)
			{
				Kernel* krnl = &(KERNEL);
				ImGui::Text("Parameter:");
				ImGui::SameLine();
				ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
				if (ImGui::BeginCombo(("##" + windowName + "_axisX").c_str(), krnl->parameters[window->orbit.xIndex].name.c_str(), 0))
				{
					for (int p = 0; p < krnl->PARAM_COUNT; p++)
					{
						if (ImGui::Selectable(krnl->parameters[p].name.c_str()))
						{
							window->orbit.xIndex = p; OrbitRedraw = true;
						}
					}

					ImGui::EndCombo();
				}

				ImGui::SameLine();
				ImGui::Text(("Variable: " + KERNEL.variables[window->variables[0]].name).c_str());
			}
			if (window->type == Metric) {
				Kernel* krnl =  &(KERNEL);


				ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.485f);
				if (ImGui::BeginCombo(("##" + windowName + "_axisX").c_str(),
					window->typeX == MDT_Variable ? krnl->variables[window->indexX].name.c_str() : krnl->parameters[window->indexX].name.c_str(), 0))
				{
					for (int v = 0; v < krnl->VAR_COUNT; v++)
					{
						if (ImGui::Selectable(krnl->variables[v].name.c_str()))
						{
							window->indexX = v;
							window->typeX = MDT_Variable;
						}
					}

					for (int p = 0; p < krnl->PARAM_COUNT; p++)
					{
						if (ImGui::Selectable(krnl->parameters[p].name.c_str()))
						{
							window->indexX = p;
							window->typeX = MDT_Parameter;
						}
					}

					ImGui::EndCombo();
				}
			}

			// Common variables
			ImPlotAxisFlags axisFlags = (toAutofit ? ImPlotAxisFlags_AutoFit : 0);
			ImPlotPlot* plot;
			ImPlot3DPlot* plot3d;
			AnalysisIndex mapIndex;
			AnalysisIndex channelMapIndex[3];
			DecaySettings* decayOfFirstIndex;
			ImPlotColormap heatmapColorMap =  ImPlotColormap_Jet;
			ImVec4 rotationEuler;
			ImVec4 rotationEulerEditable, rotationEulerBeforeEdit;
			enum PhaseSubType { Phase3D, PhaseImplot3D, Phase2D };
			PhaseSubType subtype;
			HeatmapProperties* heatmap;
			Kernel* krnl;
			Computation* cmp;

			switch (window->type)
			{
			case VarSeries:

				if (window->whiteBg) ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
				if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle, axisFlags, axisFlags))
				{
					plot = ImPlot::GetPlot(plotName.c_str());

					if (toAutofitTimeSeries)
					{
						plot->FitThisFrame = true;
						for (int i = 0; i < IMPLOT_NUM_X_AXES; i++)
						{
							ImPlotAxis& x_axis = plot->XAxis(i);
							x_axis.FitThisFrame = true;
						}
						for (int i = 0; i < IMPLOT_NUM_Y_AXES; i++)
						{
							ImPlotAxis& y_axis = plot->YAxis(i);
							y_axis.FitThisFrame = true;
						}
					}

					plot->is3d = false;

					if (computations[playedBufferIndex].ready)
					{
						uint64_t variationSize = KERNEL.VAR_COUNT * (computedSteps + 1);

						void* computedVariation = (numb*)(computations[playedBufferIndex].marshal.trajectory) + (variationSize * variation);
						memcpy(dataBuffer, computedVariation, variationSize * sizeof(numb));

						bool isTime = KERNEL.usingTime;
						float stepSize = KERNEL.GetStepSize();
						float start = !isTime ? bufferNo * KERNEL.steps + KERNEL.transientSteps : (bufferNo * KERNEL.steps + KERNEL.transientSteps) * stepSize;
						float scale = !isTime ? 1.0f : stepSize;

						if (!window->ShowMultAxes) ImPlot::SetupAxes(KERNEL.usingTime ? "Time" : "Steps", "Variable");
						else
						{
							ImPlot::SetupAxis(ImAxis_X1, KERNEL.usingTime ? "Time" : "Steps", 0);
							for (int i = 0; i < window->variableCount; i++) {
								ImPlot::SetupAxis(3 + i, KERNEL.variables[window->variables[i]].name.c_str(), 0); 
							}
						}

						for (int v = 0; v < window->variableCount; v++)
						{
							if (window->ShowMultAxes) {
								ImPlot::SetAxes(ImAxis_X1, 3 + v);
							}
							ImVec4 color;
							if (window->variableCount > 1) {
								color = ImPlot::GetColormapColor(v, window->colormap);
							}
							ImPlot::SetNextLineStyle(window->variableCount > 1 ? color : window->markerColor, window->markerWidth);
							ImPlot::PlotLine((KERNEL.variables[window->variables[v]].name + "##" + plotName + std::to_string(v)).c_str(),
								&((dataBuffer)[window->variables[v]]), computedSteps + 1, scale, start, ImPlotLineFlags_None, 0, sizeof(numb) * KERNEL.VAR_COUNT);
						}
					}

					ImPlot::EndPlot();
				}
				if (window->whiteBg) ImPlot::PopStyleColor();

				break;

			case Phase:
			case Phase2D:
				// PHASE DIAGRAM
				subtype = Phase3D;
				if (window->isImplot3d) subtype = PhaseImplot3D;
				if (window->type == Phase2D) subtype = Phase2D;

				if (subtype != Phase2D)
				{
					rotationEuler = ToEulerAngles(window->trs.quatRot);
					if (isnan(rotationEuler.x))
					{
						window->trs.quatRot = ImVec4(1.0f, 0.0f, 0.0f, 0.0f);
						rotationEuler = ToEulerAngles(window->trs.quatRot);
					}

					rotationEulerEditable = rotationEuler.Div(DEG2RAD, false);
					rotationEulerBeforeEdit = rotationEulerEditable;
					rotationEulerEditable = rotationEulerEditable.Add(window->trs.autorotate.ScalMult(frameTime, false));

					if (!window->isImplot3d && window->settingsListEnabled)
					{
						ImGui::DragFloat3("Rotation", (float*)&rotationEulerEditable, 1.0f);
						ImGui::DragFloat3("Automatic rotation", (float*)&window->trs.autorotate, 0.1f);
						ImGui::DragFloat3("Offset", (float*)&window->trs.offset, 0.01f);
						ImGui::DragFloat3("Scale", (float*)&window->trs.scale, 0.01f);
					}

					window->trs.KeepScaleAboveZero();

					if (rotationEulerBeforeEdit != rotationEulerEditable)
					{
						ImVec4 deltaEuler = rotationEulerEditable - rotationEulerBeforeEdit;
						window->trs.RotateQuaternionByEulerDrag(deltaEuler);
					}
				}

				if (window->whiteBg) 
				{ 
					ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); 
					ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
					ImPlot3D::PushStyleColor(ImPlot3DCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
				}

				axisFlags |= ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels;
				bool isPlotBegun;
				switch (subtype)
				{
				case Phase3D:
					isPlotBegun = ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, axisFlags, axisFlags);
					break;
				case PhaseImplot3D:
					isPlotBegun = ImPlot3D::BeginPlot(plotName.c_str(), ImVec2(-1, -1), ImPlot3DFlags_NoLegend | ImPlot3DFlags_NoTitle | ImPlot3DFlags_NoClip);
					break;
				case Phase2D:
					isPlotBegun = ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle, 0, 0);
					break;
				}
				
				if (isPlotBegun)
				{
					if (subtype == Phase2D) ImPlot::SetupAxes(KERNEL.variables[window->variables[0]].name.c_str(), KERNEL.variables[window->variables[1]].name.c_str());

					float plotRangeSize;
					switch (subtype)
					{
					case PhaseImplot3D:
						plot3d = ImPlot3D::GetCurrentPlot();
						break;
					case Phase3D:
					case Phase2D:
						plot = ImPlot::GetCurrentPlot();
						plotRangeSize = ((float)plot->Axes[ImAxis_X1].Range.Max - (float)plot->Axes[ImAxis_X1].Range.Min);
						if (!computations[playedBufferIndex].ready)
						{
							plotRangeSize = 10.0f;
							plot->dataMin = ImVec2(-10.0f, -10.0f);
							plot->dataMax = ImVec2(10.0f, 10.0f);
						}
					}

					float deltax = -window->trs.deltarotation.x; window->trs.deltarotation.x = 0;
					float deltay = -window->trs.deltarotation.y; window->trs.deltarotation.y = 0;

					switch (subtype)
					{
					case Phase3D:
						plot->is3d = true;
						plot->deltax = &(window->trs.deltarotation.x);
						plot->deltay = &(window->trs.deltarotation.y);
						if (deltax != 0.0f || deltay != 0.0f) addDeltaQuatRotation(window, deltax, deltay);
						rotationEuler = ToEulerAngles(window->trs.quatRot);
						populateAxisBuffer(axisBuffer, plotRangeSize / 10.0f, plotRangeSize / 10.0f, plotRangeSize / 10.0f);
						rotateOffsetBuffer(axisBuffer, 6, 3, 0, 1, 2, rotationEuler, ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 0));

						// Axis
						if (window->showAxis)
						{
							ImPlot::SetNextLineStyle(CUSTOM_COLOR(XAxis)); ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[0]), &(axisBuffer[1]), 2, 0, 0, sizeof(float) * 3);
							ImPlot::SetNextLineStyle(CUSTOM_COLOR(YAxis)); ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[6]), &(axisBuffer[7]), 2, 0, 0, sizeof(float) * 3);
							ImPlot::SetNextLineStyle(CUSTOM_COLOR(ZAxis)); ImPlot::PlotLine(plotName.c_str(), &(axisBuffer[12]), &(axisBuffer[13]), 2, 0, 0, sizeof(float) * 3);
						}

						// Axis names
						if (window->showAxisNames)
						{
							ImPlot::PushStyleColor(ImPlotCol_InlayText, CUSTOM_COLOR(XAxis)); ImPlot::PlotText(KERNEL.variables[window->variables[0]].name.c_str(), axisBuffer[0], axisBuffer[1], ImVec2(0.0f, 0.0f)); ImPlot::PopStyleColor();
							ImPlot::PushStyleColor(ImPlotCol_InlayText, CUSTOM_COLOR(YAxis)); ImPlot::PlotText(KERNEL.variables[window->variables[1]].name.c_str(), axisBuffer[6], axisBuffer[7], ImVec2(0.0f, 0.0f)); ImPlot::PopStyleColor();
							ImPlot::PushStyleColor(ImPlotCol_InlayText, CUSTOM_COLOR(ZAxis)); ImPlot::PlotText(KERNEL.variables[window->variables[2]].name.c_str(), axisBuffer[12], axisBuffer[13], ImVec2(0.0f, 0.0f)); ImPlot::PopStyleColor();
						}

						// Ruler
						if (window->showRuler)
						{
							ImVec4 scale(plotRangeSize / window->trs.scale.x, plotRangeSize / window->trs.scale.y, plotRangeSize / window->trs.scale.z, 0);
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

							DRAW_RULER_PART(CUSTOM_COLOR(XAxis).x, CUSTOM_COLOR(XAxis).y, CUSTOM_COLOR(XAxis).z, alpha0.x, scale0.x * window->trs.scale.x, scale0.x, 0);
							DRAW_RULER_PART(CUSTOM_COLOR(XAxis).x, CUSTOM_COLOR(XAxis).y, CUSTOM_COLOR(XAxis).z, alpha1.x, scale1.x * window->trs.scale.x, scale1.x, 0);

							DRAW_RULER_PART(CUSTOM_COLOR(YAxis).x, CUSTOM_COLOR(YAxis).y, CUSTOM_COLOR(YAxis).z, alpha0.y, scale0.y * window->trs.scale.y, scale0.y, 1);
							DRAW_RULER_PART(CUSTOM_COLOR(YAxis).x, CUSTOM_COLOR(YAxis).y, CUSTOM_COLOR(YAxis).z, alpha1.y, scale1.y * window->trs.scale.y, scale1.y, 1);

							DRAW_RULER_PART(CUSTOM_COLOR(ZAxis).x, CUSTOM_COLOR(ZAxis).y, CUSTOM_COLOR(ZAxis).z, alpha0.z, scale0.z * window->trs.scale.z, scale0.z, 2);
							DRAW_RULER_PART(CUSTOM_COLOR(ZAxis).x, CUSTOM_COLOR(ZAxis).y, CUSTOM_COLOR(ZAxis).z, alpha1.z, scale1.z * window->trs.scale.z, scale1.z, 2);
						}
						break;
					case PhaseImplot3D:
						//ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, ImVec4(0.07f, 0.07f, 0.07f, 1.0f));
						break;
					case Phase2D:
						plot->is3d = false;
						break;
					}

					// Drawing

					bool usePainting = colorsLUTfrom != nullptr && colorsLUTfrom->hmp.paintLUT.lut != nullptr;
					colorLUT* lut = usePainting ? &(colorsLUTfrom->hmp.paintLUT) : nullptr;

					if (computations[playedBufferIndex].ready)
					{
						if (!enabledParticles || HIRES_ON) // Trajectory - one variation, all steps
						{
							for (uint64_t drawnVariation = 0; drawnVariation < (!window->drawAllTrajectories ? 1 : computations[playedBufferIndex].marshal.totalVariations); drawnVariation++) // In case of drawing all trajectories
							{
								numb* computedVariation =
									computations[playedBufferIndex].marshal.trajectory + (computations[playedBufferIndex].marshal.variationSize * (window->drawAllTrajectories ? drawnVariation : variation));
								
								switch (subtype)
								{
								case Phase3D:
									memcpy(dataBuffer, computedVariation, computations[playedBufferIndex].marshal.variationSize * sizeof(numb));

									rotateOffsetBuffer(dataBuffer, computedSteps + 1, KERNEL.VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
										rotationEuler, window->trs.offset, window->trs.scale);
									[[fallthrough]];
								case Phase2D:
									getMinMax2D(subtype == Phase3D ? dataBuffer : computedVariation, computedSteps + 1, &(plot->dataMin), &(plot->dataMax), KERNEL.VAR_COUNT);

									if (!usePainting)
										ImPlot::SetNextLineStyle(window->plotColor);
									else
									{
										int variationGroup = getVariationGroup(lut, !window->drawAllTrajectories ? variation : drawnVariation);
										ImVec4 clr = ImPlot::SampleColormap((float)variationGroup / (lut->lutGroups - 1), colorsLUTfrom->hmp.colormap);
										clr.w = window->plotColor.w;
										ImPlot::SetNextLineStyle(clr);
									}
									ImPlot::PlotLine(plotName.c_str(),
										subtype == Phase3D ? &(dataBuffer[0]) : &(computedVariation[window->variables[0]]),
										subtype == Phase3D ? &(dataBuffer[1]) : &(computedVariation[window->variables[1]]),
										computedSteps + 1, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
									break;
								case PhaseImplot3D:
									ImPlot3D::SetupAxes(KERNEL.variables[window->variables[0]].name.c_str(), KERNEL.variables[window->variables[1]].name.c_str(), KERNEL.variables[window->variables[2]].name.c_str());

									if (!usePainting)
										ImPlot3D::SetNextLineStyle(window->plotColor);
									else
									{
										int variationGroup = getVariationGroup(lut, variation);
										ImVec4 clr = ImPlot3D::SampleColormap((float)variationGroup / (lut->lutGroups - 1), colorsLUTfrom->hmp.colormap);
										clr.w = window->plotColor.w;
										ImPlot3D::SetNextLineStyle(clr);
									}

									ImPlot3D::PlotLine(plotName.c_str(), &(computedVariation[window->variables[0]]), &(computedVariation[window->variables[1]]), &(computedVariation[window->variables[2]]),
										computedSteps + 1, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
									break;
								}
							}
						}
						else if (particleBuffer != nullptr) // Particles - all variations, one certain step
						{
							if (particleStep > KERNEL.steps) particleStep = KERNEL.steps;
							int totalVariations = computations[playedBufferIndex].marshal.totalVariations;
							int varCount = KERNEL.VAR_COUNT; // If you don't make this local, it increases the copying time by 30 times, tee-hee
							uint64_t variationSize = computations[playedBufferIndex].marshal.variationSize;
							numb* trajectory = computations[playedBufferIndex].marshal.trajectory;
							for (uint64_t v = 0; v < totalVariations; v++)
								for (int var = 0; var < varCount; var++)
									particleBuffer[v * varCount + var] = trajectory[(variationSize * v) + (varCount * particleStep) + var];

							switch (subtype)
							{
							case Phase3D:
								rotateOffsetBuffer(particleBuffer, computations[playedBufferIndex].marshal.totalVariations, KERNEL.VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
									rotationEuler, window->trs.offset, window->trs.scale);
								[[fallthrough]];
							case Phase2D:
								getMinMax2D(particleBuffer, computations[playedBufferIndex].marshal.totalVariations, &(plot->dataMin), &(plot->dataMax), KERNEL.VAR_COUNT);

								if (!usePainting)
								{
									ImPlot::SetNextLineStyle(window->markerColor);
									ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineWidth);
									ImPlot::SetNextMarkerStyle(window->markerShape, window->markerWidth);
									ImPlot::PlotScatter(plotName.c_str(),
										subtype == Phase3D ? &((particleBuffer)[0]) : &((particleBuffer)[window->variables[0]]),
										subtype == Phase3D ? &((particleBuffer)[1]) : &((particleBuffer)[window->variables[1]]),
										computations[playedBufferIndex].marshal.totalVariations, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
								}
								else if (!colorsLUTfrom->hmp.isHeatmapDirty)
								{
									for (int g = 0; g < lut->lutGroups; g++)
									{
										int lutsize = lut->lutSizes[g];

										// TODO: Look into making this step a single time, not doing it before
										for (int v = 0; v < lutsize; v++)
											for (int var = 0; var < varCount; var++)
												particleBuffer[v * varCount + var] = trajectory[(variationSize * lut->lut[g][v]) + (varCount * particleStep) + var];

										if (subtype == Phase3D)
											rotateOffsetBuffer(particleBuffer, lutsize, KERNEL.VAR_COUNT, window->variables[0], window->variables[1], window->variables[2],
												rotationEuler, window->trs.offset, window->trs.scale);

										ImVec4 clr = ImPlot::SampleColormap((float)g / (lut->lutGroups - 1), colorsLUTfrom->hmp.colormap);
										clr.w = window->markerColor.w;
										ImPlot::SetNextLineStyle(clr);
										ImPlot::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineWidth);
										ImPlot::SetNextMarkerStyle(window->markerShape, window->markerWidth);
										ImPlot::PlotScatter(plotName.c_str(), 
											subtype == Phase3D ? &((particleBuffer)[0]) : &((particleBuffer)[window->variables[0]]),
											subtype == Phase3D ? &((particleBuffer)[1]) : &((particleBuffer)[window->variables[1]]),
											lutsize, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
									}
								}
								break;
							case PhaseImplot3D:
								ImPlot3D::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineWidth);
								ImPlot3D::SetNextMarkerStyle(window->markerShape, window->markerWidth);

								if (!usePainting)
								{
									ImPlot3D::SetNextLineStyle(window->markerColor);
									ImPlot3D::PlotScatter(plotName.c_str(), &((particleBuffer)[window->variables[0]]), &((particleBuffer)[window->variables[1]]), &((particleBuffer)[window->variables[2]]),
										computations[playedBufferIndex].marshal.totalVariations, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
								}
								else if (!colorsLUTfrom->hmp.isHeatmapDirty)
								{
									for (int g = 0; g < lut->lutGroups; g++)
									{
										int lutsize = lut->lutSizes[g];
										for (int v = 0; v < lutsize; v++)
											for (int var = 0; var < varCount; var++)
												particleBuffer[v * varCount + var] = trajectory[(variationSize * lut->lut[g][v]) + (varCount * particleStep) + var];

										ImPlot3D::PushStyleVar(ImPlotStyleVar_MarkerWeight, window->markerOutlineWidth);
										ImPlot3D::SetNextMarkerStyle(window->markerShape, window->markerWidth);

										ImVec4 clr = ImPlot::SampleColormap((float)g / (lut->lutGroups - 1), ImPlotColormap_Jet);
										clr.w = window->markerColor.w;
										ImPlot3D::SetNextLineStyle(clr);
										ImPlot3D::PlotScatter(plotName.c_str(), &((particleBuffer)[window->variables[0]]), &((particleBuffer)[window->variables[1]]), &((particleBuffer)[window->variables[2]]),
											lutsize, 0, 0, sizeof(numb) * KERNEL.VAR_COUNT);
									}
								}
								break;
							}
						}
					}
					else // if computation is not ready
					{
						if (subtype == PhaseImplot3D)
						{
							float justPlotOneDot[3]{ 0.0f, 0.0f, 0.0f }; // It needs at least one dot, don't remember the details
							ImPlot3D::PlotScatter(plotName.c_str(), &(justPlotOneDot[0]), &(justPlotOneDot[1]), &(justPlotOneDot[2]), 1);
						}
					}

					// PHASE DIAGRAM END
					switch (subtype)
					{
					case PhaseImplot3D:
						//ImPlot3D::PopStyleColor(1);
						ImPlot3D::EndPlot();
						break;
					case Phase3D:
					case Phase2D:
						ImPlot::EndPlot();
						break;
					}
				}
				if (window->whiteBg)
				{
					ImPlot::PopStyleColor(2);
					ImPlot3D::PopStyleColor(1);
				}
				break;

				case Orbit:
				{
					if (window->whiteBg)
						ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));


					Kernel* krnl = &KERNEL;
					Attribute* axis = &(krnl->parameters[window->orbit.xIndex]);
					bool axisXisRanging = axis->TrueStepCount() > 1;

					if (axisXisRanging)
					{
						if (computations[playedBufferIndex].ready)
						{
							int xIndex = window->variables[0];

							int varCount = KERNEL.VAR_COUNT;
							int parCount = KERNEL.PARAM_COUNT;
							int variationSize = computations[playedBufferIndex].marshal.variationSize;

							numb stepH = krnl->stepType == 0 ? krnl->parameters[parCount - 1].values[attributeValueIndices[varCount + parCount - 1]] : (krnl->stepType == 1 ? krnl->variables[varCount - 1].values[attributeValueIndices[varCount - 1]] : (numb)1.0);

							// Buffer to hold peak data (amplitudes and indices)
							constexpr int MAX_PEAKS = 1024;
							numb* peakAmplitudes = new numb[MAX_PEAKS];
							numb* peakIntervals = new numb[MAX_PEAKS];

							numb paramStep = axis->step;
							numb paramMin = axis->min;
							uint64_t variation = 0;
							if (OrbitRedraw) { window->orbit.drawingContinuation = false; window->orbit.redrawContinuation = false; }

							if (!window->orbit.lastAttributevalueindicesContinuations.empty())
								for (int i = 0; i < varCount + parCount - 2; i++) {
									if (i != varCount + window->orbit.xIndex) {
										if (attributeValueIndices[i] != window->orbit.lastAttributevalueindicesContinuations[i]) 
										{ 
											window->orbit.drawingContinuation = false;
											window->orbit.redrawContinuation = false;
										}
									}
								}
							
							if (window->orbit.redrawContinuation || window->orbit.continuationButtonPressed) {
								window->orbit.lastAttributevalueindicesContinuations = attributeValueIndices;
								//if (window->continuationAmpsBack != NULL) {
								//    delete[]window->continuationParamIndicesBack; 
								//    delete[]window->continuationParamIndicesForward;
								//    delete[]window->continuationAmpsBack;
								//    delete[]window->continuationAmpsForward;
								//    delete[]window->continuationIntervalsBack;
								//    delete[]window->continuationParamIndicesBack;
								//}
								numb* startingVariables = new numb[varCount];
								numb* newVariables = new numb[varCount];
								numb* parameters = new numb[parCount];
								window->orbit.continuationParamIndicesBack = new numb[MAX_PEAKS * axis->stepCount];
								window->orbit.continuationParamIndicesForward = new numb[MAX_PEAKS * axis->stepCount];
								window->orbit.continuationAmpsBack = new numb[MAX_PEAKS * axis->stepCount];
								window->orbit.continuationAmpsForward = new numb[MAX_PEAKS * axis->stepCount];
								window->orbit.continuationIntervalsBack = new numb[MAX_PEAKS * axis->stepCount];
								window->orbit.continuationIntervalsForward = new numb[MAX_PEAKS * axis->stepCount];
								std::vector<numb> trajectory;
								for (int i = 0; i < varCount; i++) {
									startingVariables[i] = attributeValueIndices[i] == 0 ? KERNEL.variables[i].min : KERNEL.variables[i].min + KERNEL.variables[i].step * attributeValueIndices[i];
								}
								for (int i = 0; i < parCount; i++) {
									parameters[i] = attributeValueIndices[i + varCount] == 0 ? KERNEL.parameters[i].min : KERNEL.parameters[i].min + KERNEL.parameters[i].step * attributeValueIndices[i + varCount];
								}

								parameters[window->orbit.xIndex] = KERNEL.parameters[window->orbit.xIndex].min;
								for (int i = 0; i < KERNEL.transientSteps; i++) { kernel_FDS(selectedKernel)(startingVariables, newVariables, parameters); startingVariables = newVariables; }
								trajectory.push_back(startingVariables[xIndex]);

								int BifDotAmount = 0;

								for (int j = 0; j < axis->stepCount; j++) {
									parameters[window->orbit.xIndex] = KERNEL.parameters[window->orbit.xIndex].min + KERNEL.parameters[window->orbit.xIndex].step * j;
									for (int trajstep = 0; trajstep < variationSize / varCount; trajstep++) {
										kernel_FDS(selectedKernel)(startingVariables, newVariables, parameters);
										trajectory.push_back(newVariables[xIndex]);
										startingVariables = newVariables;
									}
									int peakCount = 0;
									bool firstpeakreached = false;
									numb temppeakindex;
									for (int trajstep = 1; trajstep < variationSize / varCount - 1 && peakCount < MAX_PEAKS; trajstep++) {
										numb prev = trajectory[trajstep - 1];
										numb curr = trajectory[trajstep];
										numb next = trajectory[trajstep + 1];
										if (curr > prev && curr > next)
										{
											if (firstpeakreached == false)
											{
												firstpeakreached = true;
												temppeakindex = (float)trajstep;
											}
											else
											{
												window->orbit.continuationAmpsForward[BifDotAmount] = curr;
												window->orbit.continuationIntervalsForward[BifDotAmount] = (trajstep - temppeakindex) * stepH;
												window->orbit.continuationParamIndicesForward[BifDotAmount] = paramMin + j * paramStep;
												temppeakindex = (float)trajstep;
												peakCount++;
												BifDotAmount++;
											}
										}
									}
									trajectory.clear();
								}
								window->orbit.bifDotAmountForward = BifDotAmount;


								for (int i = 0; i < varCount; i++) {
									startingVariables[i] = attributeValueIndices[i] == 0 ? KERNEL.variables[i].min : KERNEL.variables[i].min + KERNEL.variables[i].step * attributeValueIndices[i];
								}

								parameters[window->orbit.xIndex] = KERNEL.parameters[window->orbit.xIndex].max;
								for (int i = 0; i < KERNEL.transientSteps; i++) { kernel_FDS(selectedKernel)(startingVariables, newVariables, parameters); startingVariables = newVariables; }
								trajectory.push_back(startingVariables[xIndex]);

								BifDotAmount = 0;
								for (int j = axis->stepCount - 1; j >= 0; j--) {
									parameters[window->orbit.xIndex] = KERNEL.parameters[window->orbit.xIndex].min + KERNEL.parameters[window->orbit.xIndex].step * j;
									for (int trajstep = 0; trajstep < variationSize / varCount; trajstep++) {
										kernel_FDS(selectedKernel)(startingVariables, newVariables, parameters);
										trajectory.push_back(newVariables[xIndex]);
										startingVariables = newVariables;
									}
									int peakCount = 0;
									bool firstpeakreached = false;
									numb temppeakindex;
									for (int trajstep = 1; trajstep < variationSize / varCount - 1 && peakCount < MAX_PEAKS; trajstep++) {
										numb prev = trajectory[trajstep - 1];
										numb curr = trajectory[trajstep];
										numb next = trajectory[trajstep + 1];

										if (curr > prev && curr > next)
										{
											if (firstpeakreached == false)
											{
												firstpeakreached = true;
												temppeakindex = (float)trajstep;
											}
											else
											{
												window->orbit.continuationAmpsBack[BifDotAmount] = curr;
												window->orbit.continuationIntervalsBack[BifDotAmount] = (trajstep - temppeakindex) * stepH;
												window->orbit.continuationParamIndicesBack[BifDotAmount] = paramMin + j * paramStep;
												temppeakindex = (float)trajstep;
												peakCount++;
												BifDotAmount++;
											}
										}
									}
									trajectory.clear();
								}
								window->orbit.bifDotAmountBack = BifDotAmount;

								window->orbit.redrawContinuation = false;
								window->orbit.drawingContinuation = true;
								window->orbit.continuationButtonPressed = false;
							}

							if (window->orbit.type == OPT_Selected_Var_Section) {
								steps2Variation(&variation, &(attributeValueIndices.data()[0]), &KERNEL);

								int peakCount = 0;
								bool firstpeakreached = false;
								numb temppeakindex;
								numb* computedVariation = computations[playedBufferIndex].marshal.trajectory + (computations[playedBufferIndex].marshal.variationSize * variation);
								for (int i = 1; i < variationSize / varCount - 1 && peakCount < MAX_PEAKS; i++)
								{
									numb prev = computedVariation[xIndex + varCount * i - varCount];
									numb curr = computedVariation[xIndex + varCount * i];
									numb next = computedVariation[xIndex + varCount * i + varCount];
									if (curr > prev && curr > next)
									{
										if (firstpeakreached == false)
										{
											firstpeakreached = true;
											temppeakindex = (float)i;
										}
										else
										{

											peakAmplitudes[peakCount] = curr;
											peakIntervals[peakCount] = (i - temppeakindex) * stepH;
											peakCount++;
											temppeakindex = (float)i;
										}
									}
								}

								numb minX = peakIntervals[0], maxX = peakIntervals[0];
								numb minY = peakAmplitudes[0], maxY = peakAmplitudes[0];
								for (int i = 0; i < peakCount - 1; ++i)
								{
									if (peakIntervals[i] < minX) minX = peakIntervals[i];
									if (peakIntervals[i] > maxX) maxX = peakIntervals[i];
									if (peakAmplitudes[i + 1] < minY) minY = peakAmplitudes[i + 1];
									if (peakAmplitudes[i + 1] > maxY) maxY = peakAmplitudes[i + 1];
								}
								std::vector<numb> SliceForwardInt; std::vector<numb> SliceForwardPeak;
								std::vector<numb> SliceBackwardInt; std::vector<numb>  SliceBackwardPeak;
								int forNum = 0; int backNum = 0;
								bool sectionFound = false;
								if (window->orbit.drawingContinuation && window->orbit.continuationAmpsForward != NULL || window->orbit.lastAttributevalueindicesContinuations != attributeValueIndices && window->orbit.continuationAmpsForward != NULL) {
									SliceBackwardInt.clear(); SliceBackwardPeak.clear(); SliceForwardInt.clear(); SliceForwardPeak.clear();
									window->orbit.lastAttributevalueindicesContinuations = attributeValueIndices;
									for (int i = 0; i < window->orbit.bifDotAmountForward - 1; i++) {
										if (window->orbit.continuationParamIndicesForward[i] == axis->min + attributeValueIndices[varCount + window->orbit.xIndex] * axis->step) {
											if (!sectionFound) { sectionFound = true; }
											SliceForwardInt.push_back(window->orbit.continuationIntervalsForward[i]); SliceForwardPeak.push_back(window->orbit.continuationAmpsForward[i]);
											forNum++;
										}
									}
									for (int i = 0; i < window->orbit.bifDotAmountBack - 1; i++) {
										if (window->orbit.continuationParamIndicesBack[i] == axis->min + attributeValueIndices[varCount + window->orbit.xIndex] * axis->step) {
											if (!sectionFound) { sectionFound = true; }
											SliceBackwardInt.push_back(window->orbit.continuationIntervalsBack[i]); SliceBackwardPeak.push_back(window->orbit.continuationAmpsBack[i]);
											backNum++;
										}
									}
								}
								if (ImPlot::BeginPlot(("##" + plotName + "_ChosenVariation").c_str(), window->orbit.invertedAxes ? "Peaks" : "Intervals", window->orbit.invertedAxes ? "Intervals" : "Peaks", ImVec2(-1, -1), ImPlotFlags_NoTitle, 0, 0)) {
									plot = ImPlot::GetPlot(("##" + plotName + "_ChosenVariation").c_str()); plot->is3d = false;
									ImPlot::SetupAxisLimits(ImAxis_X1, minX * 0.95f, maxX * 1.05f, ImGuiCond_None);
									ImPlot::SetupAxisLimits(ImAxis_Y1, minY * 0.95f, maxY * 1.05f, ImGuiCond_None);
									ImPlot::SetupFinish();
									ImPlot::SetNextMarkerStyle(window->markerShape, window->orbit.pointSize, window->plotColor, -1.0, window->plotColor);
									if (!window->orbit.invertedAxes) {
										ImPlot::PlotScatter(window->orbit.drawingContinuation ? ("Standard##" + plotName + "_ChosenVariationPlot").c_str() : ("##" + plotName + "_ChosenVariationPlot").c_str(), peakIntervals, peakAmplitudes, peakCount - 1);
										if (window->orbit.drawingContinuation && window->orbit.continuationAmpsForward != NULL) {
											ImPlot::SetNextMarkerStyle(window->orbit.dotShapeForward, window->orbit.pointSizeForward, window->orbit.dotColorForward, IMPLOT_AUTO, window->orbit.dotColorForward);
											ImPlot::PlotScatter(("Forward##" + plotName + "_ChosenVariationPlot").c_str(), SliceForwardInt.data(), SliceForwardPeak.data(), forNum - 1);
											ImPlot::SetNextMarkerStyle(window->orbit.dotShapeBack, window->orbit.pointSizeBack, window->orbit.dotColorBack, IMPLOT_AUTO, window->orbit.dotColorBack);
											ImPlot::PlotScatter(("Backward##" + plotName + "_ChosenVariationPlot").c_str(), SliceBackwardInt.data(), SliceBackwardPeak.data(), backNum - 1);
										}
									}
									else {
										ImPlot::PlotScatter(("##" + plotName + "_ChosenVariationPlot").c_str(), peakAmplitudes, peakIntervals, peakCount - 1);
									}
									;
									ImPlot::EndPlot();
								}
							}
							else {
								if (OrbitRedraw) { window->orbit.areValuesDirty = OrbitRedraw; }
								if (window->orbit.lastAttributeValueIndices.size() != 0) {
									for (int i = 0; i < varCount + parCount - 2; i++) {
										if (i != varCount + window->orbit.xIndex) {
											if (attributeValueIndices[i] != window->orbit.lastAttributeValueIndices[i]) window->orbit.areValuesDirty = true;
										}
									}
								}
								if (window->orbit.areValuesDirty) {
									if (window->orbit.bifAmps != nullptr) { delete[]window->orbit.bifAmps; delete[]window->orbit.bifIntervals; delete[]window->orbit.bifParamIndices; }
									window->orbit.bifAmps = new numb[MAX_PEAKS * axis->stepCount];
									window->orbit.bifIntervals = new numb[MAX_PEAKS * axis->stepCount];
									window->orbit.bifParamIndices = new numb[MAX_PEAKS * axis->stepCount];
									std::vector<int> tempattributeValueIndices = attributeValueIndices;
									window->orbit.lastAttributeValueIndices = attributeValueIndices;
									int BifDotAmount = 0;
									for (int j = 0; j < axis->stepCount; j++)
									{
										tempattributeValueIndices[window->orbit.xIndex + varCount] = j;
										steps2Variation(&variation, &(tempattributeValueIndices.data()[0]), &KERNEL);
										numb* computedVariation = computations[playedBufferIndex].marshal.trajectory + (computations[playedBufferIndex].marshal.variationSize * variation);
										int peakCount = 0;
										bool firstpeakreached = false;
										numb temppeakindex;
										for (int i = 1; i < variationSize / varCount - 1 && peakCount < MAX_PEAKS; i++)
										{
											numb prev = computedVariation[xIndex + varCount * i - varCount];
											numb curr = computedVariation[xIndex + varCount * i];
											numb next = computedVariation[xIndex + varCount * i + varCount];
											if (curr > prev && curr > next)
											{
												if (firstpeakreached == false)
												{
													firstpeakreached = true;
													temppeakindex = (float)i;
												}
												else
												{
													window->orbit.bifAmps[BifDotAmount] = curr;
													window->orbit.bifIntervals[BifDotAmount] = (i - temppeakindex) * stepH;
													window->orbit.bifParamIndices[BifDotAmount] = paramMin + j * paramStep;
													temppeakindex = (float)i;
													peakCount++;
													BifDotAmount++;

												}
											}
										}

									}
									window->orbit.bifDotAmount = BifDotAmount;
									window->orbit.areValuesDirty = false;
								}
								numb minX = window->orbit.bifParamIndices[0], maxX = window->orbit.bifParamIndices[0];
								numb minY = window->orbit.bifAmps[0], maxY = window->orbit.bifAmps[0];
								numb minZ = window->orbit.bifIntervals[0]; numb maxZ = window->orbit.bifIntervals[0];
								if (window->orbit.type == OPT_Peak_Bifurcation) {

									for (int i = 0; i < window->orbit.bifDotAmount - 1; ++i)
									{
										maxX = window->orbit.bifParamIndices[i];
										if (window->orbit.bifAmps[i + 1] < minY) minY = window->orbit.bifAmps[i + 1];
										if (window->orbit.bifAmps[i + 1] > maxY) maxY = window->orbit.bifAmps[i + 1];
									}
									if (ImPlot::BeginPlot((plotName + "_BifDiagrams").c_str(), window->orbit.invertedAxes ? "Peaks" : axis->name.c_str(), window->orbit.invertedAxes ? axis->name.c_str() : "Peaks", ImVec2(-1, -1), ImPlotFlags_NoTitle, 0, 0)) {
										plot = ImPlot::GetPlot((plotName + "_BifDiagrams").c_str()); plot->is3d = false;
										ImPlot::SetupAxisLimits(ImAxis_X1, minX * 0.95f, maxX * 1.05f, ImGuiCond_None);
										ImPlot::SetupAxisLimits(ImAxis_Y1, minY * 0.95f, maxY * 1.05f, ImGuiCond_None);
										ImPlot::SetupFinish();
										ImPlot::SetNextMarkerStyle(window->markerShape, window->orbit.pointSize, window->plotColor, IMPLOT_AUTO, window->plotColor);
										if (!window->orbit.invertedAxes) {
											ImPlot::PlotScatter(window->orbit.drawingContinuation ? ("Standard##Peak to Parameter " + plotName).c_str() : ("##Peak to Parameter " + plotName).c_str(), window->orbit.bifParamIndices, window->orbit.bifAmps, window->orbit.bifDotAmount);
											if (window->orbit.drawingContinuation) {
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeForward, window->orbit.pointSizeForward, window->orbit.dotColorForward, IMPLOT_AUTO, window->orbit.dotColorForward);
												ImPlot::PlotScatter(("Forward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationParamIndicesForward, window->orbit.continuationAmpsForward, window->orbit.bifDotAmountForward);
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeBack, window->orbit.pointSizeBack, window->orbit.dotColorBack, IMPLOT_AUTO, window->orbit.dotColorBack);
												ImPlot::PlotScatter(("Backward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationParamIndicesBack, window->orbit.continuationAmpsBack, window->orbit.bifDotAmountBack);
											}
										}
										else {
											ImPlot::PlotScatter(window->orbit.drawingContinuation ? ("Standard##Peak to Parameter " + plotName).c_str() : ("##Peak to Parameter " + plotName).c_str(), window->orbit.bifAmps, window->orbit.bifParamIndices, window->orbit.bifDotAmount);
											if (window->orbit.drawingContinuation) {
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeForward, window->orbit.pointSizeForward, window->orbit.dotColorForward, IMPLOT_AUTO, window->orbit.dotColorForward);
												ImPlot::PlotScatter(("Forward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationAmpsForward, window->orbit.continuationParamIndicesForward, window->orbit.bifDotAmountForward);
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeBack, window->orbit.pointSizeBack, window->orbit.dotColorBack, IMPLOT_AUTO, window->orbit.dotColorBack);
												ImPlot::PlotScatter(("Backward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationAmpsBack, window->orbit.continuationParamIndicesBack, window->orbit.bifDotAmountBack);
											}
										}
										if (ImGui::IsMouseDown(0) && ImGui::IsKeyPressed(ImGuiMod_Shift) && ImGui::IsMouseHoveringRect(plot->PlotRect.Min, plot->PlotRect.Max) && plot->ContextLocked || plot->shiftClicked) {
											numb MousePosX;
											window->orbit.invertedAxes ? MousePosX = (numb)ImPlot::GetPlotMousePos().y : MousePosX = (numb)ImPlot::GetPlotMousePos().x;
											if (axis->min > MousePosX)attributeValueIndices[window->orbit.xIndex + varCount] = 0;
											else if (axis->max < MousePosX)attributeValueIndices[window->orbit.xIndex + varCount] = axis->stepCount - 1;
											else {
												numb NotRoundedIndex = (MousePosX - paramMin) / (axis->max - paramMin) * axis->stepCount;
												int index = static_cast<int>(std::round(NotRoundedIndex)); if (index > axis->stepCount - 1)index = axis->stepCount - 1;
												attributeValueIndices[window->orbit.xIndex + varCount] = index;
											}
										}
										if (plot->shiftSelected) {
											kernelNew.parameters[window->orbit.xIndex].min = plot->shiftSelect1Location.x;
											kernelNew.parameters[window->orbit.xIndex].max = plot->shiftSelect2Location.x;
											if (window->orbit.isAutoComputeOn) computeAfterShiftSelect = true;
										}
										if (window->orbit.showParLines) {
											double value = attributeValueIndices[varCount + window->orbit.xIndex] * paramStep + paramMin;
											if (!window->orbit.invertedAxes)ImPlot::DragLineX(0, &value, window->orbit.markerColor, window->orbit.markerWidth, ImPlotDragToolFlags_NoInputs);
											else ImPlot::DragLineY(0, &value, window->orbit.markerColor, window->orbit.markerWidth, ImPlotDragToolFlags_NoInputs);
										}

										ImPlot::EndPlot();
									}
								}
								else if (window->orbit.type == OPT_Interval_Bifurcation) {
									minY = window->orbit.bifIntervals[0], maxY = window->orbit.bifIntervals[0];
									for (int i = 0; i < window->orbit.bifDotAmount - 1; ++i)
									{
										maxX = window->orbit.bifParamIndices[i];
										if (window->orbit.bifIntervals[i + 1] < minY) minY = window->orbit.bifIntervals[i + 1];
										if (window->orbit.bifIntervals[i + 1] > maxY) maxY = window->orbit.bifIntervals[i + 1];
									}
									if (ImPlot::BeginPlot((plotName + "_BifAmp").c_str(), window->orbit.invertedAxes ? "Intervals" : axis->name.c_str(), window->orbit.invertedAxes ? axis->name.c_str() : "Intervals", ImVec2(-1, -1), ImPlotFlags_NoTitle, 0, 0)) {
										plot = ImPlot::GetPlot((plotName + "_BifAmp").c_str()); plot->is3d = false;
										ImPlot::SetupAxisLimits(ImAxis_X1, minX * 0.95f, maxX * 1.05f, ImGuiCond_None);
										ImPlot::SetupAxisLimits(ImAxis_Y1, minY * 0.95f, maxY * 1.05f, ImGuiCond_None);
										ImPlot::SetupFinish();

										ImPlot::SetNextMarkerStyle(window->markerShape, window->orbit.pointSize, window->plotColor, IMPLOT_AUTO, window->plotColor);
										if (!window->orbit.invertedAxes) {
											ImPlot::PlotScatter(window->orbit.drawingContinuation ? ("Standard##Interval to Parameter " + plotName).c_str() : ("##Interval to Parameter " + plotName).c_str(), window->orbit.bifParamIndices, window->orbit.bifIntervals, window->orbit.bifDotAmount);
											if (window->orbit.drawingContinuation) {
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeForward, window->orbit.pointSizeForward, window->orbit.dotColorForward, IMPLOT_AUTO, window->orbit.dotColorForward);
												ImPlot::PlotScatter(("Forward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationParamIndicesForward, window->orbit.continuationIntervalsForward, window->orbit.bifDotAmountForward);
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeBack, window->orbit.pointSizeBack, window->orbit.dotColorBack, IMPLOT_AUTO, window->orbit.dotColorBack);
												ImPlot::PlotScatter(("Backward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationParamIndicesBack, window->orbit.continuationIntervalsBack, window->orbit.bifDotAmountBack);
											}
										}
										else {
											ImPlot::PlotScatter(window->orbit.drawingContinuation ? ("Standard##Interval to Parameter " + plotName).c_str() : ("##Interval to Parameter " + plotName).c_str(), window->orbit.bifIntervals, window->orbit.bifParamIndices, window->orbit.bifDotAmount);
											if (window->orbit.drawingContinuation) {
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeForward, window->orbit.pointSizeForward, window->orbit.dotColorForward, IMPLOT_AUTO, window->orbit.dotColorForward);
												ImPlot::PlotScatter(("Forward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationIntervalsForward, window->orbit.continuationParamIndicesForward, window->orbit.bifDotAmountForward);
												ImPlot::SetNextMarkerStyle(window->orbit.dotShapeBack, window->orbit.pointSizeBack, window->orbit.dotColorBack, IMPLOT_AUTO, window->orbit.dotColorBack);
												ImPlot::PlotScatter(("Backward continuation##Peak to Parameter " + plotName).c_str(), window->orbit.continuationIntervalsBack, window->orbit.continuationParamIndicesBack, window->orbit.bifDotAmountBack);
											}
										}

										if (ImGui::IsMouseDown(0) && ImGui::IsKeyPressed(ImGuiMod_Shift) && ImGui::IsMouseHoveringRect(plot->PlotRect.Min, plot->PlotRect.Max) && plot->ContextLocked || plot->shiftClicked) {
											numb MousePosX;
											window->orbit.invertedAxes ? MousePosX = (numb)ImPlot::GetPlotMousePos().y : MousePosX = (numb)ImPlot::GetPlotMousePos().x;
											if (axis->min > MousePosX)attributeValueIndices[window->orbit.xIndex + varCount] = 0;
											else if (axis->max < MousePosX)attributeValueIndices[window->orbit.xIndex + varCount] = axis->stepCount - 1;
											else {
												numb NotRoundedIndex = (MousePosX - paramMin) / (axis->max - paramMin) * axis->stepCount;
												int index = static_cast<int>(std::round(NotRoundedIndex)); if (index > axis->stepCount - 1)index = axis->stepCount - 1;
												attributeValueIndices[window->orbit.xIndex + varCount] = index;
											}
										}

										if (plot->shiftSelected) {
											kernelNew.parameters[window->orbit.xIndex].min = plot->shiftSelect1Location.x;
											kernelNew.parameters[window->orbit.xIndex].max = plot->shiftSelect2Location.x;
											if (window->orbit.isAutoComputeOn) computeAfterShiftSelect = true;
										}
										if (window->orbit.showParLines) {
											double value = attributeValueIndices[varCount + window->orbit.xIndex] * paramStep + paramMin;
											if (!window->orbit.invertedAxes)ImPlot::DragLineX(0, &value, window->orbit.markerColor, window->orbit.markerWidth, ImPlotDragToolFlags_NoInputs);
											else ImPlot::DragLineY(0, &value, window->orbit.markerColor, window->orbit.markerWidth, ImPlotDragToolFlags_NoInputs);
										}
										ImPlot::EndPlot();
									}
								}
								else if (window->orbit.type == OPT_Bifurcation_3D) {
									if (ImPlot3D::BeginPlot((plotName + "_Bif3D").c_str(), ImVec2(-1, -1), ImPlotFlags_NoTitle)) {
										ImPlot3DContext* TheContext = ImPlot3D::GImPlot3D;
										plot3d = TheContext->CurrentPlot;
										plot3d->RangeMax().y;
										static float  xs[4], ys[4], zs[4];

										ys[0] = plot3d->RangeMin().y; ys[1] = plot3d->RangeMin().y; ys[2] = plot3d->RangeMax().y; ys[3] = plot3d->RangeMax().y;  zs[0] = plot3d->RangeMin().z; zs[1] = plot3d->RangeMax().z; zs[2] = plot3d->RangeMax().z; zs[3] = plot3d->RangeMin().z; xs[0] = paramMin + paramStep * attributeValueIndices[window->orbit.xIndex + krnl->VAR_COUNT]; xs[3] = xs[2] = xs[1] = xs[0];
										ImPlot3D::SetupAxis(ImAxis3D_X, axis->name.c_str());
										ImPlot3D::SetupAxis(ImAxis3D_Y, "Peaks");
										ImPlot3D::SetupAxis(ImAxis3D_Z, "Intervals");
										ImPlot3D::SetNextMarkerStyle(window->markerShape, window->orbit.pointSize, window->plotColor, IMPLOT_AUTO, window->plotColor);
										ImPlot3D::PlotScatter(window->orbit.drawingContinuation ? ("Normal##Bif#d_Diagram" + plotName).c_str() : ("##Bif#d_Diagram" + plotName).c_str(), window->orbit.bifParamIndices, window->orbit.bifAmps, window->orbit.bifIntervals, window->orbit.bifDotAmount, 0, 0);
										if (window->orbit.drawingContinuation) {
											ImPlot3D::SetNextMarkerStyle(window->orbit.dotShapeForward, window->orbit.pointSizeForward, window->orbit.dotColorForward, IMPLOT_AUTO, window->orbit.dotColorForward);
											ImPlot3D::PlotScatter(("Forward continuation##3d" + plotName).c_str(), window->orbit.continuationParamIndicesForward, window->orbit.continuationAmpsForward, window->orbit.continuationIntervalsForward, window->orbit.bifDotAmountForward, 0, 0);
											ImPlot3D::SetNextMarkerStyle(window->orbit.dotShapeBack, window->orbit.pointSizeBack, window->orbit.dotColorBack, IMPLOT_AUTO, window->orbit.dotColorBack);
											ImPlot3D::PlotScatter(("Backward continuation##3d " + plotName).c_str(), window->orbit.continuationParamIndicesBack, window->orbit.continuationAmpsBack, window->orbit.continuationIntervalsBack, window->orbit.bifDotAmountBack, 0, 0);
										}
										ImPlot3D::SetNextFillStyle(window->orbit.markerColor);
										ImPlot3D::SetNextLineStyle(window->orbit.markerColor, 2);
										ImPlot3D::PlotQuad("", &xs[0], &ys[0], &zs[0], 4);
										ImPlot3D::EndPlot();
									}
								}
							}
							delete[]peakAmplitudes; delete[]peakIntervals;
						}
					}
					else
					{
						if (!axisXisRanging) ImGui::Text(("Parameter " + axis->name + " is fixed").c_str());
					}

					if (window->whiteBg)
						ImPlot::PopStyleColor();

				}
				break;

				case Metric:
					{
					if (window->whiteBg)
						ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
						
						Kernel* krnl =  &(KERNEL);
						Computation* cmp =  &(computations[playedBufferIndex]);
						Attribute* axis = window->typeX == MDT_Variable ? &(krnl->variables[window->indexX]) : &(krnl->parameters[window->indexX]);
						bool axisIsRanging = axis->TrueStepCount() > 1;
						if (axisIsRanging) {
							//numb minX, stepX, maxX;
							//int xSize;
							
							HeatmapProperties* hmp = nullptr;
							uint64_t variation=0;
							if (ImPlot::BeginPlot(("##Metric_Plot" + plotName).c_str(), ImVec2(-1, -1), ImPlotFlags_NoTitle)) {
								
								plot = ImPlot::GetPlot(("##Metric_Plot" + plotName).c_str());
								plot->is3d = false;
								//ImPlot::SetNextLineStyle(window->markerColor, window->markerWidth);
								
								if (!window->ShowMultAxes) 
									ImPlot::SetupAxes(window->typeX == MDT_Variable ? krnl->variables[window->indexX].name.c_str() : krnl->parameters[window->indexX].name.c_str(), "Index"); 
								else
								{
									ImPlot::SetupAxis(ImAxis_X1, window->typeX == MDT_Variable ? krnl->variables[window->indexX].name.c_str() : krnl->parameters[window->indexX].name.c_str(), 0);
									for (int i = 0; i < window->variableCount; i++) 
									{
										ImPlot::SetupAxis(3+i, indices[(AnalysisIndex)window->variables[i]].name.c_str(), 0);
									}
								}
								for (int j = 0; j < window->variableCount; j++) {
									mapIndex = (AnalysisIndex)window->variables[j];
									numb* MapSlice = cmp->marshal.maps + index2port(cmp->marshal.kernel.analyses, mapIndex)->offset * cmp->marshal.totalVariations;
									numb* Xaxis = new numb[axis->stepCount];
									numb* Yaxis = new numb[axis->stepCount];
									std::vector<int> tempattributeValueIndices = attributeValueIndices;
									for (int i = 0; i < axis->stepCount; i++) {
										if (window->typeX == MDT_Variable)tempattributeValueIndices[window->indexX] = i;
										else tempattributeValueIndices[window->indexX + krnl->VAR_COUNT] = i;
										steps2Variation(&variation, &(tempattributeValueIndices.data()[0]), &KERNEL);
										Xaxis[i] = axis->min + axis->step * i;
										Yaxis[i] = MapSlice[variation];
									}
									if (window->ShowMultAxes) {
										ImPlot::SetAxes(ImAxis_X1, 3+j);
									}
									ImVec4 color;
									if (window->variableCount > 1) {
										color = ImPlot::GetColormapColor(j, window->colormap);
									}
									ImPlot::SetNextLineStyle(window->variableCount>1 ? color : window->markerColor,window->markerWidth);
									ImPlot::PlotLine(indices[(AnalysisIndex)window->variables[j]].name.c_str(), Xaxis, Yaxis, axis->stepCount);
									delete[] Xaxis;
									delete[] Yaxis;
									
								}
								if (ImGui::IsMouseDown(0) && ImGui::IsKeyPressed(ImGuiMod_Shift) && ImGui::IsMouseHoveringRect(plot->PlotRect.Min, plot->PlotRect.Max) && plot->ContextLocked || plot->shiftClicked) {
									numb MousePosX = (numb)ImPlot::GetPlotMousePos().x;
									if (axis->min > MousePosX)window->typeX == MDT_Variable ? attributeValueIndices[window->indexX] = 0 : attributeValueIndices[window->indexX + krnl->VAR_COUNT] = 0;
									else if (axis->max < MousePosX)window->typeX == MDT_Variable ? attributeValueIndices[window->indexX] = axis->stepCount - 1 : attributeValueIndices[window->indexX + krnl->VAR_COUNT] = axis->stepCount - 1;
									else {
										numb NotRoundedIndex = (MousePosX - axis->min) / (axis->max - axis->min) * axis->stepCount;
										int index = static_cast<int>(std::round(NotRoundedIndex)); if (index > axis->stepCount - 1)index = axis->stepCount - 1;
										window->typeX == MDT_Variable ? attributeValueIndices[window->indexX] = index : attributeValueIndices[window->indexX + krnl->VAR_COUNT] = index;
									}
								}
								if (plot->shiftSelected) {
									kernelNew.parameters[window->orbit.xIndex].min = plot->shiftSelect1Location.x;
									kernelNew.parameters[window->orbit.xIndex].max = plot->shiftSelect2Location.x;
									if (window->orbit.isAutoComputeOn) computeAfterShiftSelect = true;
								}
								if (window->orbit.showParLines) {
									double value = axis->min + axis->step * attributeValueIndices[window->typeX == MDT_Variable ? window->indexX : window->indexX + krnl->VAR_COUNT];
									ImPlot::DragLineX(0, &value, window->orbit.markerColor, window->orbit.markerWidth, ImPlotDragToolFlags_NoInputs);
								}
								ImPlot::EndPlot();
							}
							
							
						}
						else {
							ImGui::Text(("Axis " + axis->name + " is fixed").c_str());
						}
						if (window->whiteBg)
							ImPlot::PopStyleColor();
					}
					break;

				case Decay:
					if (window->decay.settings.size() == 0) for (int i = 0; i < window->variables.size(); i++) window->decay.settings.push_back(DecaySettings());
					if (window->decay.thresholdNames.size() == 0) for (int i = 0; i < window->variables.size(); i++) window->decay.thresholdNames.push_back("Threshold");

					window->decay.ForceDecayThresholdCount();

					mapIndex    = (AnalysisIndex)window->variables[0]; // Using only the first index for previews etc., but editing all indices of the window
					heatmap     = &window->hmp;
					cmp         = &(computations[1 - bufferToFillIndex]);
					krnl        = &(KERNEL);
					decayOfFirstIndex = &(window->decay.settings[0]);

					ImGui::Text("Source:"); ImGui::SameLine();
					if (ImGui::RadioButton(("Index##DTSIndex" + plotName).c_str(), decayOfFirstIndex->source == DTS_Index))
						for (int i : window->variables) indices[(AnalysisIndex)i].decay.source = DTS_Index;
					ImGui::SameLine();
					if (ImGui::RadioButton(("Delta##DTSDelta" + plotName).c_str(), decayOfFirstIndex->source == DTS_Delta))
						for (int i : window->variables) indices[(AnalysisIndex)i].decay.source = DTS_Delta;

					ImGui::Text("Mode:"); ImGui::SameLine();
					if (ImGui::RadioButton(("Less than##DTMLess" + plotName).c_str(), decayOfFirstIndex->mode == DTM_Less))
						for (int i : window->variables) indices[(AnalysisIndex)i].decay.mode = DTM_Less;
					ImGui::SameLine();
					if (ImGui::RadioButton(("More than##DTMMore" + plotName).c_str(), decayOfFirstIndex->mode == DTM_More))
						for (int i : window->variables) indices[(AnalysisIndex)i].decay.mode = DTM_More;
					ImGui::SameLine();
					if (ImGui::RadioButton(("Absolute value more than##DTMAbsMore" + plotName).c_str(), decayOfFirstIndex->mode == DTM_Abs_More))
						for (int i : window->variables) indices[(AnalysisIndex)i].decay.mode = DTM_Abs_More;

					// Threshold input

					if (ImGui::BeginCombo(("##ThresholdsTableCombo" + plotName).c_str(), "Edit thresholds"))
					{
						int thresholdsCount = window->decay.thresholdCount;
						int indicesCount = (int)window->variables.size();

						ImGui::BeginDisabled(thresholdsCount < 2);
						ImGui::SameLine();
						if (ImGui::Button(("Remove threshold##minusThreshold" + plotName).c_str()))
						{
							window->decay.thresholdCount--;
							window->decay.DeleteBuffers();
						};
						ImGui::EndDisabled();
						ImGui::SameLine();
						if (ImGui::Button(("Add threshold##plusThreshold" + plotName).c_str()))
						{
							window->decay.thresholdCount++;
							window->decay.DeleteBuffers();
						};

						if (ImGui::BeginTable(("##ThresholdsTable" + plotName).c_str(), 1 + indicesCount))
						{
							// Columns setup
							for (int c = 0; c < 1 + indicesCount; c++) ImGui::TableSetupColumn(nullptr);

							// Headers
							ImGui::TableNextRow();
							ImGui::TableSetColumnIndex(0);
							ImGui::Text("Name        ");
							for (int i = 0; i < indicesCount; i++)
							{
								ImGui::TableSetColumnIndex(i + 1);
								std::string name = indices[(AnalysisIndex)window->variables[i]].name;
								name.resize(16);
								ImGui::Text(name.c_str());
							}

							// Thresholds
							for (int t = 0; t < thresholdsCount; t++)
							{
								ImGui::TableNextRow();
								ImGui::TableSetColumnIndex(0);

								char nameBuffer[33];
								std::strncpy(nameBuffer, window->decay.thresholdNames[t].c_str(), 32);
								nameBuffer[32] = '\0';
								ImGui::SetNextItemWidth(-1);
								if (ImGui::InputText(("##ThresholdName" + std::to_string(t) + plotName).c_str(), nameBuffer, 32)) window->decay.thresholdNames[t] = std::string(nameBuffer);

								// Indeces
								for (int i = 0; i < indicesCount; i++)
								{
									ImGui::TableSetColumnIndex(i + 1);
									ImGui::SetNextItemWidth(-1);
									ImGui::InputFloat(("##DT" + std::to_string(t) + " " + std::to_string(i) + plotName).c_str(), &(window->decay.settings[i].thresholds[t]));
								}
							}

							ImGui::EndTable();
						}
						ImGui::EndCombo();
					}

					window->decay.ForceDecayThresholdCount();

					for (int i = 0; i < window->variables.size(); i++) 
						indices[(AnalysisIndex)window->variables[i]].decay.thresholds = window->decay.settings[i].thresholds;

					if (cmp->marshal.indecesDelta == nullptr) TEXT_AND_BREAK("Decay plot delta not ready yet")
					if (!cmp->calculateDeltaDecay) TEXT_AND_BREAK("Delta and decay calculation is disabled")

					if (heatmap->areValuesDirty)
					{
						window->decay.RecalculatePlot(cmp, window->variables, heatmap->values.mapValueIndex);
						heatmap->areValuesDirty = false;
					}

					if (window->decay.calcLifetime) ImGui::Text(("Avg lifetime: " + std::to_string(window->decay.lifetime)).c_str());

					if (cmp->isFirst && !window->isYLog)
					{
						ImPlot::SetNextAxisLimits(ImAxis_Y1, 0.0, (double)cmp->marshal.totalVariations, ImPlotCond_Always);
						window->decay.markerPosition = !KERNEL.usingTime ? KERNEL.transientSteps : KERNEL.transientTime;
					}

					if (window->decay.buffer.size() > 0 && window->decay.buffer[0].size() > 0)
					{
						ImPlot::PushStyleColor(ImPlotCol_PlotBg, window->decay.plotFillColor);
						if (ImPlot::BeginPlot(("##Decay_Plot" + plotName).c_str(), ImVec2(-1, -1), ImPlotFlags_NoTitle))
						{
							ImPlot::SetupAxes(!KERNEL.usingTime ? "Steps" : "Time", "Variations alive", toAutofitTimeSeries ? ImPlotAxisFlags_AutoFit : 0, 0);
							ImPlot::SetupAxisScale(ImAxis_Y1, !window->isYLog ? ImPlotScale_Linear : ImPlotScale_Log10);

							plot = ImPlot::GetPlot(("##Decay_Plot" + plotName).c_str());
							plot->is3d = false;

							if (toAutofitTimeSeries)
							{
								plot->FitThisFrame = true;
								for (int i = 0; i < IMPLOT_NUM_X_AXES; i++)
								{
									ImPlotAxis& x_axis = plot->XAxis(i);
									x_axis.FitThisFrame = true;
								}
							}

							ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, window->decay.fillAlpha);
							for (int t = 0; t < window->decay.thresholdCount; t++)
							{
								ImPlot::SetNextFillStyle(window->decay.thresholdCount > 1 ? ImPlot::GetColormapColor(t, window->colormap) : window->plotColor);
								ImPlot::PlotShaded(((window->decay.thresholdCount > 1 ? window->decay.thresholdNames[t] : "") + "##" + plotName + "_plot" + std::to_string(t)).c_str(),
									&(window->decay.buffer[t][0]), &(window->decay.alive[t][0]), (int)window->decay.buffer[t].size(), (double)cmp->marshal.totalVariations);
							}
							ImPlot::PopStyleVar();
							for (int t = 0; t < window->decay.thresholdCount; t++)
							{
								ImPlot::SetNextLineStyle(window->decay.thresholdCount > 1 ? ImPlot::GetColormapColor(t, window->colormap) : window->plotColor);
								ImPlot::PlotLine(((window->decay.thresholdCount > 1 ? window->decay.thresholdNames[t] : "") + "##" + plotName + "_plot" + std::to_string(t)).c_str(),
									&(window->decay.buffer[t][0]), &(window->decay.alive[t][0]), (int)window->decay.buffer[t].size());
							}

							if (window->decay.calcLifetime)
							{
								double prevMarkerPosition = window->decay.markerPosition;
								ImPlot::DragLineX(0, &(window->decay.markerPosition), window->markerColor, window->markerWidth);
								if (window->decay.markerPosition != prevMarkerPosition || toAutofitTimeSeries) window->decay.RecalculateLifetime(cmp);
							}

							ImPlot::EndPlot();
						}
						ImPlot::PopStyleColor();
					}
					else
					{
						ImGui::Text("Decay plot not ready yet");
					}
					break;

				case Heatmap:
				case MCHeatmap:
				{
					bool isMC = window->type == MCHeatmap; // Is multi-channel (RGB)
					mapIndex = (AnalysisIndex)window->variables[0];
					if (isMC) for (int ch = 0; ch < 3; ch++) channelMapIndex[ch] = (AnalysisIndex)window->variables[ch];
					bool isHires = window->isTheHiresWindow(hiresIndex) || window->isFrozenAsHires;
					HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;
					Kernel* krnl = isHires ? &kernelHiresComputed : &(KERNEL);
					Computation* cmp = isHires ? &computationHires : &(computations[playedBufferIndex]);
					std::vector<int>* avi = isHires ? &attributeValueIndicesHires : &attributeValueIndices;
					uint64_t* var = isHires ? &variationHires : &variation;
					uint64_t* prevVar = isHires ? &prevVariationHires : &prevVariation;
					bool hiresFrozenAndAvailable = window->isFrozenAsHires && window->hmp.values.valueBuffer != nullptr;
					bool showLegend = heatmap->showLegend;

					if (!isMC)
					{
						if (!indices[mapIndex].enabled && !hiresFrozenAndAvailable) TEXT_AND_BREAK(("Index " + indices[mapIndex].name + " has been disabled").c_str())
						if (!index2port(cmp->marshal.kernel.analyses, mapIndex)->used && !hiresFrozenAndAvailable) TEXT_AND_BREAK(("Index " + indices[mapIndex].name + " has not been computed").c_str())

						if (window->deltaState == DS_Delta && !cmp->marshal.indecesDeltaExists) TEXT_AND_BREAK("Delta not computed yet")
						if (window->deltaState == DS_Decay && !cmp->marshal.indecesDeltaExists) TEXT_AND_BREAK("Delta not computed yet")
						if (window->deltaState == DS_Delta && cmp->bufferNo < 2) TEXT_AND_BREAK("Delta not ready yet")
						if (window->deltaState == DS_Decay && cmp->bufferNo < 2) TEXT_AND_BREAK("Delta not ready yet")
					}

					Attribute* axisX = heatmap->typeX == MDT_Variable ? &(krnl->variables[heatmap->indexX]) : &(krnl->parameters[heatmap->indexX]);
					Attribute* axisY = heatmap->typeY == MDT_Variable ? &(krnl->variables[heatmap->indexY]) : &(krnl->parameters[heatmap->indexY]);

					if (axisX->TrueStepCount() <= 1)    TEXT_AND_BREAK(("Axis " + axisX->name + " is fixed").c_str())
					if (axisY->TrueStepCount() <= 1)    TEXT_AND_BREAK(("Axis " + axisY->name + " is fixed").c_str())
					if (axisX == axisY)                 TEXT_AND_BREAK("X and Y axis are the same")
					if (!window->deltaState == DS_No && !cmp->calculateDeltaDecay) TEXT_AND_BREAK("Delta and decay calculation is disabled")

					if (ImGui::BeginTable((plotName + "_table").c_str(), showLegend ? 2 : 1, ImGuiTableFlags_Reorderable, ImVec2(-1, 0)))
					{
						axisFlags = 0;

						ImGui::TableSetupColumn(nullptr);
						if (showLegend) ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, 170.0f * (isMC ? 3 : 1));
						ImGui::TableNextRow();

						numb min = 0.0f, max = 0.0f;

						ImGui::TableSetColumnIndex(0);
						ImPlot::PushColormap(heatmapColorMap);
						ImVec2 plotSize;

						HeatmapSizing sizing;
						if (window->whiteBg) { ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); }
						if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend, axisFlags, axisFlags))
						{
							plot = ImPlot::GetPlot(plotName.c_str());
							plot->is3d = false;
							plot->isHeatmapSelectionModeOn = heatmap->isHeatmapSelectionModeOn;

							if (cmp->ready || hiresFrozenAndAvailable)
							{
								sizing.loadPointers(krnl, heatmap);
								sizing.initValues();

								if (heatmap->initClickedLocation)
								{
									heatmap->lastClickedLocation = ImVec2((float)sizing.minX, (float)sizing.minY);
									window->dragLineHiresPos = ImVec2((float)sizing.minX, (float)sizing.minY);
									heatmap->initClickedLocation = false;
								}

								if (!window->isFrozen)
								{
									if (heatmap->showActualDiapasons)
									{
										// Values
										heatmap->lastClickedLocation.x = (float)valueFromStep(sizing.minX, sizing.stepX,
											(*avi)[sizing.hmp->indexX + (sizing.hmp->typeX == MDT_Variable ? 0 : krnl->VAR_COUNT)]);
										heatmap->lastClickedLocation.y = (float)valueFromStep(sizing.minY, sizing.stepY,
											(*avi)[sizing.hmp->indexY + (sizing.hmp->typeY == MDT_Variable ? 0 : krnl->VAR_COUNT)]);
									}
									else
									{
										// Steps
										heatmap->lastClickedLocation.x = (float)(*avi)[sizing.hmp->indexX + (sizing.hmp->typeX == MDT_Variable ? 0 : krnl->VAR_COUNT)];
										heatmap->lastClickedLocation.y = (float)(*avi)[sizing.hmp->indexY + (sizing.hmp->typeY == MDT_Variable ? 0 : krnl->VAR_COUNT)];
									}

									// Hovering
									if (heatmap->valueDisplay == VDM_Always || heatmap->valueDisplay == VDM_Split)
									{
										int stepX = 0;
										int stepY = 0;

										if (heatmap->showActualDiapasons)
										{
											// Values
											stepX = stepFromValue(sizing.minX, sizing.stepX, plot->mouseLocation.x);
											stepY = stepFromValue(sizing.minY, sizing.stepY, plot->mouseLocation.y);
										}
										else
										{
											// Steps
											stepX = (int)floor(plot->mouseLocation.x);
											stepY = (int)floor(plot->mouseLocation.y);
										}

										if (stepX < 0 || stepX >= (sizing.hmp->typeX == MDT_Variable ? krnl->variables[sizing.hmp->indexX].TrueStepCount() : krnl->parameters[sizing.hmp->indexX].TrueStepCount())
											|| stepY < 0 || stepY >= (sizing.hmp->typeY == MDT_Variable ? krnl->variables[sizing.hmp->indexY].TrueStepCount() : krnl->parameters[sizing.hmp->indexY].TrueStepCount()))
										{
											heatmap->values.hoveredAway = true;
											for (int ch = 0; ch < 3; ch++)
												heatmap->channel[ch].hoveredAway = true;
										}
										else
										{
											heatmap->values.hoveredAway = false;
											for (int ch = 0; ch < 3; ch++)
												heatmap->channel[ch].hoveredAway = false;

											if (!isMC)
											{
												if (heatmap->values.valueBuffer != nullptr)
												{
													heatmap->values.lastHovered = heatmap->values.valueBuffer[sizing.xSize * stepY + stepX];
													//heatmap->values.lastShiftClicked = heatmap->values.valueBuffer[sizing.xSize * stepY + stepX];
												}
											}
											else
											{
												for (int ch = 0; ch < 3; ch++)
												{
													if (heatmap->channel[ch].valueBuffer != nullptr)
													{
														heatmap->channel[ch].lastHovered = heatmap->channel[ch].valueBuffer[sizing.xSize * stepY + stepX];
														//heatmap->channel[ch].lastShiftClicked = heatmap->channel[ch].valueBuffer[sizing.xSize * stepY + stepX];
													}
												}
											}
										}
									}

									// Choosing configuration
									if (plot->shiftClicked && plot->shiftClickLocation.x != 0.0)
									{
										int stepX = 0;
										int stepY = 0;

										if (heatmap->showActualDiapasons)
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

#define IGNOREOUTOFREACH    if (stepX < 0 || stepX >= (sizing.hmp->typeX == MDT_Variable ? krnl->variables[sizing.hmp->indexX].TrueStepCount() : krnl->parameters[sizing.hmp->indexX].TrueStepCount())) break; \
						if (stepY < 0 || stepY >= (sizing.hmp->typeY == MDT_Variable ? krnl->variables[sizing.hmp->indexY].TrueStepCount() : krnl->parameters[sizing.hmp->indexY].TrueStepCount())) break;

										switch (sizing.hmp->typeX)
										{
										case MDT_Variable:
											IGNOREOUTOFREACH;
											(*avi)[sizing.hmp->indexX] = stepX;
											heatmap->lastClickedLocation.x = plot->shiftClickLocation.x;
											break;
										case MDT_Parameter:
											IGNOREOUTOFREACH;
											(*avi)[krnl->VAR_COUNT + sizing.hmp->indexX] = stepX;
											heatmap->lastClickedLocation.x = plot->shiftClickLocation.x;
											break;
										}

										switch (sizing.hmp->typeY)
										{
										case MDT_Variable:
											IGNOREOUTOFREACH;
											(*avi)[sizing.hmp->indexY] = stepY;
											heatmap->lastClickedLocation.y = plot->shiftClickLocation.y;
											break;
										case MDT_Parameter:
											IGNOREOUTOFREACH;
											(*avi)[krnl->VAR_COUNT + sizing.hmp->indexY] = stepY;
											heatmap->lastClickedLocation.y = plot->shiftClickLocation.y;
											break;
										}
									}

									if (heatmap->valueDisplay == VDM_OnlyOnShiftClick || heatmap->valueDisplay == VDM_Split)
									{
										int stepX = 0;
										int stepY = 0;

										stepX = (*avi)[sizing.hmp->indexX + (sizing.hmp->typeX == MDT_Variable ? 0 : krnl->VAR_COUNT)];
										stepY = (*avi)[sizing.hmp->indexY + (sizing.hmp->typeY == MDT_Variable ? 0 : krnl->VAR_COUNT)];

										if (stepX < 0 || stepX >= (sizing.hmp->typeX == MDT_Variable ? krnl->variables[sizing.hmp->indexX].TrueStepCount() : krnl->parameters[sizing.hmp->indexX].TrueStepCount())
											|| stepY < 0 || stepY >= (sizing.hmp->typeY == MDT_Variable ? krnl->variables[sizing.hmp->indexY].TrueStepCount() : krnl->parameters[sizing.hmp->indexY].TrueStepCount())) {
										}
										else
										{
											if (!isMC)
											{
												if (heatmap->values.valueBuffer != nullptr)
												{
													//heatmap->values.lastShiftClicked = heatmap->values.valueBuffer[sizing.xSize * stepY + stepX];
													heatmap->values.lastSelected = heatmap->values.valueBuffer[sizing.xSize * stepY + stepX];
												}
											}
											else
											{
												for (int ch = 0; ch < 3; ch++)
												{
													if (heatmap->channel[ch].valueBuffer != nullptr)
													{
														//heatmap->channel[ch].lastShiftClicked = heatmap->channel[ch].valueBuffer[sizing.xSize * stepY + stepX];
														heatmap->channel[ch].lastSelected = heatmap->channel[ch].valueBuffer[sizing.xSize * stepY + stepX];
													}
												}
											}
										}
									}

									if (isHires)
									{
										if (plot->shiftClicked && plot->shiftClickLocation.x != 0.0)
										{
											numb valueX = 0.0; numb valueY = 0.0;

											if (heatmap->showActualDiapasons)
											{
												// Values
												valueX = plot->shiftClickLocation.x;
												valueY = plot->shiftClickLocation.y;
												window->dragLineHiresPos = ImVec2((float)valueX, (float)valueY);
											}
											else
											{
												// Steps
												valueX = valueFromStep(sizing.minX, sizing.stepX, (int)floor(plot->shiftClickLocation.x));
												valueY = valueFromStep(sizing.minY, sizing.stepY, (int)floor(plot->shiftClickLocation.y));
												window->dragLineHiresPos = ImVec2(floor(plot->shiftClickLocation.x), floor(plot->shiftClickLocation.y));
											}

											hiresShiftClickCompute();
										}
									}

									if (plot->shiftSelected)
									{
										heatmapRangingSelection(window, plot, &sizing, isHires);
									}
								}

								sizing.initCutoff((float)plot->Axes[plot->CurrentX].Range.Min, (float)plot->Axes[plot->CurrentY].Range.Min,
									(float)plot->Axes[plot->CurrentX].Range.Max, (float)plot->Axes[plot->CurrentY].Range.Max, heatmap->showActualDiapasons);
								if (autofitHeatmap || toAutofit)
								{
									plot->Axes[plot->CurrentX].Range.Min = sizing.mapX1;
									plot->Axes[plot->CurrentX].Range.Max = sizing.mapX2;
									plot->Axes[plot->CurrentY].Range.Min = sizing.mapY1;
									plot->Axes[plot->CurrentY].Range.Max = sizing.mapY2;
								}

								// Drawing

								int mapSize = sizing.xSize * sizing.ySize;
								if (heatmap->lastBufferSize != mapSize && !window->isFrozen)
								{
									if (!isMC)
									{
										if (heatmap->values.valueBuffer != nullptr)
										{
											delete[] heatmap->values.valueBuffer;
											heatmap->values.valueBuffer = nullptr;
										}
									}
									else
									{
										for (int ch = 0; ch < 3; ch++)
											if (heatmap->channel[ch].valueBuffer != nullptr)
											{
												delete[] heatmap->channel[ch].valueBuffer;
												heatmap->channel[ch].valueBuffer = nullptr;
											}
									}

									if (heatmap->pixelBuffer != nullptr) delete[] heatmap->pixelBuffer;
									if (heatmap->indexBuffer != nullptr) delete[] heatmap->indexBuffer;

									heatmap->lastBufferSize = mapSize;

									if (!isMC)
									{
										heatmap->values.valueBuffer = new numb[mapSize];
									}
									else
									{
										for (int ch = 0; ch < 3; ch++)
										{
											//if (cmp->marshal.kernel.mapDatas[channelMapIndex[ch]].toCompute)
											heatmap->channel[ch].valueBuffer = new numb[mapSize];
										}
									}

									heatmap->pixelBuffer = new unsigned char[mapSize * 4];
									heatmap->indexBuffer = new int[cmp->marshal.totalVariations];
									heatmap->areValuesDirty = true;
								}

								if (*var != *prevVar) heatmap->areValuesDirty = true;

								if (heatmap->areValuesDirty)
								{
									if (!window->isFrozen)
									{
										if (!isMC)
										{
											extractMap((window->deltaState == DS_No ? cmp->marshal.maps : (window->deltaState == DS_Delta ? cmp->marshal.indecesDelta : cmp->marshal.indecesDecay))
												+ (index2port(cmp->marshal.kernel.analyses, mapIndex)->offset + heatmap->values.mapValueIndex) * cmp->marshal.totalVariations,
												heatmap->values.valueBuffer, heatmap->indexBuffer, &(avi->data()[0]),
												sizing.hmp->typeX == MDT_Parameter ? sizing.hmp->indexX + krnl->VAR_COUNT : sizing.hmp->indexX,
												sizing.hmp->typeY == MDT_Parameter ? sizing.hmp->indexY + krnl->VAR_COUNT : sizing.hmp->indexY,
												krnl);
										}
										else
										{
											for (int c = 0; c < 3; c++)
											{
												if (channelMapIndex[c] == -1) continue;
												if (!index2port(cmp->marshal.kernel.analyses, channelMapIndex[c])->used) continue;

												extractMap(cmp->marshal.maps + (index2port(cmp->marshal.kernel.analyses, channelMapIndex[c])->offset + heatmap->channel[c].mapValueIndex) * cmp->marshal.totalVariations,
													heatmap->channel[c].valueBuffer, heatmap->indexBuffer, &(avi->data()[0]),
													sizing.hmp->typeX == MDT_Parameter ? sizing.hmp->indexX + krnl->VAR_COUNT : sizing.hmp->indexX,
													sizing.hmp->typeY == MDT_Parameter ? sizing.hmp->indexY + krnl->VAR_COUNT : sizing.hmp->indexY,
													krnl);
											}
										}
									}

									heatmap->areValuesDirty = false;
									heatmap->isHeatmapDirty = true;
								}

								if (!isMC && !heatmap->values.areHeatmapLimitsDefined && heatmap->values.valueBuffer != nullptr)
								{
									if (!heatmap->ignoreNextLimitsRecalculation)
										getMinMax(heatmap->values.valueBuffer, sizing.xSize * sizing.ySize, &heatmap->values.heatmapMin, &heatmap->values.heatmapMax);

									heatmap->ignoreNextLimitsRecalculation = false;
									heatmap->values.areHeatmapLimitsDefined = true;
								}
								if (isMC)
								{
									for (int ch = 0; ch < 3; ch++)
										if (!heatmap->channel[ch].areHeatmapLimitsDefined && window->variables[ch] > -1 && heatmap->channel[ch].valueBuffer != nullptr)
										{
											if (!heatmap->ignoreNextLimitsRecalculation)
												getMinMax(heatmap->channel[ch].valueBuffer, sizing.xSize * sizing.ySize, &heatmap->channel[ch].heatmapMin, &heatmap->channel[ch].heatmapMax);

											heatmap->ignoreNextLimitsRecalculation = false;
											heatmap->channel[ch].areHeatmapLimitsDefined = true;
										}
								}

								// Do not reload values when variating map axes (map values don't change anyway)
								if (*var != *prevVar) heatmap->isHeatmapDirty = true;

								// Image init
								if (heatmap->isHeatmapDirty && !window->isFrozen)
								{
									if (!isMC)
										MapToImg(heatmap->values.valueBuffer, &(heatmap->pixelBuffer), sizing.xSize, sizing.ySize, heatmap->values.heatmapMin, heatmap->values.heatmapMax, heatmap->colormap);
									else
										MultichannelMapToImg(heatmap, &(heatmap->pixelBuffer), sizing.xSize, sizing.ySize, channelMapIndex[0] > -1, channelMapIndex[1] > -1, channelMapIndex[2] > -1);

									heatmap->isHeatmapDirty = false;

									// COLORS
									if (!isMC)
									{
										heatmap->paintLUT.Clear();

										heatmap->paintLUT.lutGroups = paintLUTsize;
										heatmap->paintLUT.lut = new int* [paintLUTsize];
										for (int i = 0; i < paintLUTsize; i++) heatmap->paintLUT.lut[i] = new int[cmp->marshal.totalVariations];
										heatmap->paintLUT.lutSizes = new int[paintLUTsize];

										setupLUT((window->deltaState == DS_No ? cmp->marshal.maps : (window->deltaState == DS_Delta ? cmp->marshal.indecesDelta : cmp->marshal.indecesDecay))
											+ (index2port(cmp->marshal.kernel.analyses, mapIndex)->offset + heatmap->values.mapValueIndex) * cmp->marshal.totalVariations,
											cmp->marshal.totalVariations, heatmap->paintLUT.lut, heatmap->paintLUT.lutSizes, paintLUTsize,
											heatmap->values.heatmapMin, heatmap->values.heatmapMax);
									}

									releaseHeatmap(window, isHires);
								}

								if (heatmap->texture == nullptr)
								{
									bool ret = LoadTextureFromRaw(&(heatmap->pixelBuffer), sizing.xSize, sizing.ySize, (ID3D11ShaderResourceView**)&(heatmap->texture), g_pd3dDevice);
									IM_ASSERT(ret);
								}

								ImPlotPoint from = heatmap->showActualDiapasons ? ImPlotPoint(sizing.minX, sizing.maxY + sizing.stepY) : ImPlotPoint(0, sizing.ySize);
								ImPlotPoint to = heatmap->showActualDiapasons ? ImPlotPoint(sizing.maxX + sizing.stepX, sizing.minY) : ImPlotPoint(sizing.xSize, 0);

								ImPlot::SetupAxes(sizing.hmp->typeX == MDT_Parameter ? krnl->parameters[sizing.hmp->indexX].name.c_str() : krnl->variables[sizing.hmp->indexX].name.c_str(),
									sizing.hmp->typeY == MDT_Parameter ? krnl->parameters[sizing.hmp->indexY].name.c_str() : krnl->variables[sizing.hmp->indexY].name.c_str());

								ImPlot::PlotImage(("Map " + std::to_string(mapIndex) + "##" + plotName + std::to_string(0)).c_str(), (ImTextureID)(heatmap->texture),
									from, to, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

								// If shift-clicked inside of plot object in correct window
								bool plotDragShiftClicked = ImGui::IsMouseDown(0) && ImGui::IsKeyPressed(ImGuiMod_Shift) 
									&& ImGui::IsMouseHoveringRect(plot->PlotRect.Min, plot->PlotRect.Max) && plot->ContextLocked && !isHires;
								if (plotDragShiftClicked) {
									//values
									if (heatmap->showActualDiapasons) {
										numb MousePosX = (numb)ImPlot::GetPlotMousePos().x; //find mouse Position
										if (axisX->min > MousePosX)sizing.hmp->typeX == MDT_Variable ? attributeValueIndices[sizing.hmp->indexX] = 0 : attributeValueIndices[sizing.hmp->indexX + krnl->VAR_COUNT] = 0; // check if position is under minimum of parameter
										else if (axisX->max < MousePosX)sizing.hmp->typeX == MDT_Variable ? attributeValueIndices[sizing.hmp->indexX] = axisX->stepCount - 1 : attributeValueIndices[sizing.hmp->indexX + krnl->VAR_COUNT] = axisX->stepCount - 1; // or above maximum
										else {
											numb NotRoundedIndex = (MousePosX - axisX->min - axisX->step) / (axisX->max - axisX->min) * axisX->stepCount;   // mathematically find what index would the mouse pos be
											int index = static_cast<int>(std::round(NotRoundedIndex)); index < 0 ? index = 0 : 1; if (index > axisX->stepCount - 1)index = axisX->stepCount - 1;    // round the index and check if it is in parameter bounds
											sizing.hmp->typeX == MDT_Variable ? attributeValueIndices[sizing.hmp->indexX] = index : attributeValueIndices[sizing.hmp->indexX + krnl->VAR_COUNT] = index;    // write index into attrivuteValueIndices
										}
										numb MousePosY = (numb)ImPlot::GetPlotMousePos().y;
										if (axisY->min > MousePosY)sizing.hmp->typeY == MDT_Variable ? attributeValueIndices[sizing.hmp->indexY] = 0 : attributeValueIndices[sizing.hmp->indexY + krnl->VAR_COUNT] = 0;
										else if (axisY->max < MousePosY)sizing.hmp->typeY == MDT_Variable ? attributeValueIndices[sizing.hmp->indexY] = axisY->stepCount - 1 : attributeValueIndices[sizing.hmp->indexY + krnl->VAR_COUNT] = axisY->stepCount - 1;
										else {
											numb NotRoundedIndex = (MousePosY - axisY->min - axisY->step) / (axisY->max - axisY->min) * axisY->stepCount;
											int index = static_cast<int>(std::round(NotRoundedIndex)); index < 0 ? index = 0 : 1; if (index > axisY->stepCount - 1)index = axisY->stepCount - 1;
											sizing.hmp->typeY == MDT_Variable ? attributeValueIndices[sizing.hmp->indexY] = index : attributeValueIndices[sizing.hmp->indexY + krnl->VAR_COUNT] = index;
										}
									}
									//steps
									else {
										numb MousePosX = (numb)ImPlot::GetPlotMousePos().x - 0.5;
										if (0 > MousePosX)sizing.hmp->typeX == MDT_Variable ? attributeValueIndices[sizing.hmp->indexX] = 0 : attributeValueIndices[sizing.hmp->indexX + krnl->VAR_COUNT] = 0;
										else if (axisX->stepCount < MousePosX)sizing.hmp->typeX == MDT_Variable ? attributeValueIndices[sizing.hmp->indexX] = axisX->stepCount - 1 : attributeValueIndices[sizing.hmp->indexX + krnl->VAR_COUNT] = axisX->stepCount - 1;
										else {
											int index = static_cast<int>(std::round(MousePosX)); index < 0 ? index = 0 : 1; if (index > axisX->stepCount - 1)index = axisX->stepCount - 1;
											sizing.hmp->typeX == MDT_Variable ? attributeValueIndices[sizing.hmp->indexX] = index : attributeValueIndices[sizing.hmp->indexX + krnl->VAR_COUNT] = index;
										}
										numb MousePosY = (numb)ImPlot::GetPlotMousePos().y - 0.5;
										if (0 > MousePosY)sizing.hmp->typeY == MDT_Variable ? attributeValueIndices[sizing.hmp->indexY] = 0 : attributeValueIndices[sizing.hmp->indexY + krnl->VAR_COUNT] = 0;
										else if (axisY->stepCount < MousePosY)sizing.hmp->typeY == MDT_Variable ? attributeValueIndices[sizing.hmp->indexY] = axisY->stepCount - 1 : attributeValueIndices[sizing.hmp->indexY + krnl->VAR_COUNT] = axisY->stepCount - 1;
										else {
											int index = static_cast<int>(std::round(MousePosY)); index < 0 ? index = 0 : 1; if (index > axisY->stepCount - 1)index = axisY->stepCount - 1;
											sizing.hmp->typeY == MDT_Variable ? attributeValueIndices[sizing.hmp->indexY] = index : attributeValueIndices[sizing.hmp->indexY + krnl->VAR_COUNT] = index;
										}

									}
								}

								// Value labels
								if (sizing.cutWidth > 0 && sizing.cutHeight > 0) // If there's anything to be shown in the plot
								{
									if (heatmap->showHeatmapValues)
									{
										if (!isMC)
										{
											void* cutoffHeatmap = new numb[sizing.cutHeight * sizing.cutWidth];
											cutoff2D(heatmap->values.valueBuffer, (numb*)cutoffHeatmap,
												sizing.xSize, sizing.ySize, sizing.cutMinX, sizing.cutMinY, sizing.cutMaxX, sizing.cutMaxY);

											ImPlot::PlotHeatmap(("MapLabels " + std::to_string(mapIndex) + "##" + plotName + std::to_string(0)).c_str(),
												(numb*)cutoffHeatmap, sizing.cutHeight, sizing.cutWidth, -1234.0, 1.0, "%.3f",
												ImPlotPoint(sizing.mapX1Cut, sizing.mapY1Cut), ImPlotPoint(sizing.mapX2Cut, sizing.mapY2Cut));

											delete[] cutoffHeatmap;
										}
										else
										{
											void* cutoffHeatmap;
											for (int ch = 0; ch < 3; ch++)
											{
												if (window->variables[ch] == -1) continue;

												std::string mapName = indices[(AnalysisIndex)window->variables[ch]].name;

												cutoffHeatmap = new numb[sizing.cutHeight * sizing.cutWidth];
												cutoff2D(heatmap->channel[ch].valueBuffer, (numb*)cutoffHeatmap,
													sizing.xSize, sizing.ySize, sizing.cutMinX, sizing.cutMinY, sizing.cutMaxX, sizing.cutMaxY);

												ImPlot::PlotHeatmap(("MapLabels_" + std::to_string(mapIndex) + "##" + plotName + "_ch" + std::to_string(ch)).c_str(),
													(numb*)cutoffHeatmap, sizing.cutHeight, sizing.cutWidth, -1234.0, 1.0,
													(ch == 0 ? mapName + ": %.3f\n \n " : (ch == 1 ? " \n" + mapName + ": %.3f\n " : " \n \n" + mapName + ": %.3f")).c_str(),
													ImPlotPoint(sizing.mapX1Cut, sizing.mapY1Cut), ImPlotPoint(sizing.mapX2Cut, sizing.mapY2Cut));

												delete[] cutoffHeatmap;
											}
										}
									}
								}

								if (heatmap->showDragLines)
								{
									double valueX = (double)heatmap->lastClickedLocation.x + (heatmap->showActualDiapasons ? sizing.stepX * 0.5 : 0.5);
									double valueY = (double)heatmap->lastClickedLocation.y + (heatmap->showActualDiapasons ? sizing.stepY * 0.5 : 0.5);

									ImPlot::DragLineX(0, &valueX, window->markerColor, window->markerWidth, ImPlotDragToolFlags_NoInputs);
									ImPlot::DragLineY(1, &valueY, window->markerColor, window->markerWidth, ImPlotDragToolFlags_NoInputs);
								}

								// Show value in the bottom-left corner
								if (heatmap->valueDisplay != VDM_Never)
								{
									if (!isMC)
									{
										if (heatmap->values.valueBuffer != nullptr)
										{
											std::string heatmapValueString = std::to_string(
												heatmap->valueDisplay == VDM_OnlyOnShiftClick ? heatmap->values.lastSelected : heatmap->values.lastHovered);
											if (heatmap->values.hoveredAway) heatmapValueString = "";
											//std::string heatmapValueString = std::to_string(heatmap->values.lastShiftClicked);
											ImVec2 valueTextSize = ImGui::CalcTextSize(heatmapValueString.c_str());
											ImDrawList* draw_list = ImPlot::GetPlotDrawList();
											ImVec2 text_pos = ImVec2(ImPlot::GetPlotPos().x + 10, ImPlot::GetPlotPos().y + ImPlot::GetPlotSize().y - valueTextSize.y - 10);
											draw_list->AddText(text_pos, ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_Text)), heatmapValueString.c_str());
										}
									}
									else
									{
										int shift = 0;
										for (int ch = 0; ch < 3; ch++)
											if (heatmap->channel[ch].valueBuffer != nullptr) shift++;

										for (int ch = 0; ch < 3; ch++)
										{
											if (heatmap->channel[ch].valueBuffer != nullptr)
											{
												std::string heatmapValueString = std::to_string(
													heatmap->valueDisplay == VDM_OnlyOnShiftClick ? heatmap->channel[ch].lastSelected : heatmap->channel[ch].lastHovered);
												if (heatmap->channel[ch].hoveredAway) heatmapValueString = "";
												//std::string heatmapValueString = std::to_string(heatmap->channel[ch].lastShiftClicked);
												ImVec2 valueTextSize = ImGui::CalcTextSize(heatmapValueString.c_str());
												ImDrawList* draw_list = ImPlot::GetPlotDrawList();
												ImVec2 text_pos = ImVec2(ImPlot::GetPlotPos().x + 10, ImPlot::GetPlotPos().y + ImPlot::GetPlotSize().y - (valueTextSize.y * shift) - 10);
												draw_list->AddText(text_pos, ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_Text)), heatmapValueString.c_str());
												shift--;
											}
										}
									}
								}

								plotSize = ImPlot::GetPlotSize();
							}

							ImPlot::EndPlot();
						}

						ImPlot::PopColormap();

						if (showLegend)
						{
							ImGui::TableSetColumnIndex(1);
							if (isMC)
							{
								if (ImGui::BeginTable((plotName + "_rgbColormapsTable").c_str(), 3, ImGuiTableFlags_Reorderable, ImVec2(-1, 0)))
								{
									// Column settings
									for (int ch = 0; ch < 3; ch++)
									{
										ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthFixed, 160.0f);
									}

									// Map choice
									ImGui::TableNextRow();
									for (int ch = 0; ch < 3; ch++)
									{
										ImGui::TableSetColumnIndex(ch);
										int prevChannel = window->variables[ch];

										ImGui::SetNextItemWidth(160.0f);
										mapSelectionCombo("##" + window->name + "_channel" + std::to_string(ch), window->variables[ch], true);

										if (window->variables[ch] != prevChannel)
										{
											heatmap->areValuesDirty = true;
											heatmap->channel[ch].areHeatmapLimitsDefined = false;
										}
									}

									// Map value choice
									bool needMapValueRow = false;
									for (int ch = 0; ch < 3; ch++)
									{
										if (window->variables[ch] == -1) continue;
										if (index2port(KERNEL.analyses, (AnalysisIndex)window->variables[ch])->size > 1)
										{
											needMapValueRow = true;
											break;
										}
									}
									if (needMapValueRow)
									{
										ImGui::TableNextRow();
										for (int ch = 0; ch < 3; ch++)
										{
											ImGui::TableSetColumnIndex(ch);
											int prevValueIndex = heatmap->channel[ch].mapValueIndex;

											ImGui::SetNextItemWidth(160.0f);
											mapValueSelectionCombo((AnalysisIndex)window->variables[ch], ch, window->name, heatmap);

											if (prevValueIndex != heatmap->channel[ch].mapValueIndex)
											{
												heatmap->areValuesDirty = true;
												heatmap->channel[ch].areHeatmapLimitsDefined = false;
											}
										}
									}

									// Colormaps and minmaxes
									ImGui::TableNextRow();
									for (int ch = 0; ch < 3; ch++)
									{
										ImGui::TableSetColumnIndex(ch);

										HeatmapValues* values = &(heatmap->channel[ch]);
										numb minBefore = values->heatmapMin, maxBefore = values->heatmapMax;

										float heatMinFloat = values->heatmapMin, heatMaxFloat = values->heatmapMax;
										std::string maxName = "Max##Max" + std::to_string(ch);
										std::string minName = "Min##Min" + std::to_string(ch);
										std::string colormapName = "##HeatScale_" + std::to_string(ch);

										ImGui::SetNextItemWidth(120);
										ImGui::DragFloat(maxName.c_str(), &heatMaxFloat, 0.01f);
										ImU32 markerColor32 = ImGui::ColorConvertFloat4ToU32(window->markerColor);
										//ColormapMarkerSettings markerSettings(&(values->lastShiftClicked), 1, markerColor32, window->markerWidth);
										ColormapMarkerSettings markerSettings(heatmap->valueDisplay != VDM_Never,
											&(heatmap->valueDisplay == VDM_Always ? values->lastHovered : values->lastSelected),
											1, markerColor32, window->markerWidth);
										ImPlot::ColormapScale(colormapName.c_str(), values->heatmapMin, values->heatmapMax, markerSettings, ImVec2(120, plotSize.y - 30.0f), "%g", 0, multichannelHeatmapColormaps[ch]);
										ImGui::SetNextItemWidth(120);
										ImGui::DragFloat(minName.c_str(), &heatMinFloat, 0.01f);

										values->heatmapMin = (numb)heatMinFloat; values->heatmapMax = (numb)heatMaxFloat;

										if (minBefore != values->heatmapMin || maxBefore != values->heatmapMax) heatmap->isHeatmapDirty = true;
									}

									ImGui::EndTable();
								}
							}
							else
							{
								HeatmapValues* values = &(heatmap->values);
								numb minBefore = values->heatmapMin, maxBefore = values->heatmapMax;

								float heatMinFloat = values->heatmapMin, heatMaxFloat = values->heatmapMax;
								std::string maxName = "Max##Max";
								std::string minName = "Min##Min";
								std::string colormapName = "##HeatScale";

								ImGui::SetNextItemWidth(120);
								ImGui::DragFloat(maxName.c_str(), &heatMaxFloat, 0.01f);
								ImU32 markerColor32 = ImGui::ColorConvertFloat4ToU32(window->markerColor);
								ColormapMarkerSettings markerSettings(heatmap->valueDisplay != VDM_Never,
									&(heatmap->valueDisplay == VDM_Always ? values->lastHovered : values->lastSelected),
									1, markerColor32, window->markerWidth);
								//ColormapMarkerSettings markerSettings(&(values->lastShiftClicked), 1, markerColor32, window->markerWidth);
								ImPlot::ColormapScale(colormapName.c_str(), values->heatmapMin, values->heatmapMax, markerSettings, ImVec2(120, plotSize.y - 30.0f), "%g", 0, heatmap->colormap);
								ImGui::SetNextItemWidth(120);
								ImGui::DragFloat(minName.c_str(), &heatMinFloat, 0.01f);

								values->heatmapMin = (numb)heatMinFloat; values->heatmapMax = (numb)heatMaxFloat;

								if (minBefore != values->heatmapMin || maxBefore != values->heatmapMax) heatmap->isHeatmapDirty = true;
							}
						}

						if (window->whiteBg) ImPlot::PopStyleColor(2);

						ImGui::EndTable();
					}
				}
					break;

				case IndSeries:
				{
					if (window->whiteBg)
						ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

					Kernel* krnl = &(KERNEL);
					Computation* cmp = &(computations[playedBufferIndex]);
					Marshal* mrsl = &cmp->marshal;

					if (!cmp->marshal.maps) TEXT_AND_BREAK("No variables/parameters ranging")
					if (indSeriesReset) { window->indSeries.clear(); indSeriesReset = false; }
					if (cmp->bufferNo != window->prevbufferNo)
					{
						window->prevbufferNo = cmp->bufferNo;
						for (int ind = 0; ind < window->variableCount; ind++)
						{
							uint64_t variation;
							steps2Variation(&variation, &(attributeValueIndices.data()[0]), &KERNEL);
							mapIndex = (AnalysisIndex)window->variables[ind];
							numb* MapSlice = cmp->marshal.maps + index2port(cmp->marshal.kernel.analyses, mapIndex)->offset * cmp->marshal.totalVariations;
							window->indSeries.push_back(MapSlice[variation]);
						}
					}

					if (window->prevbufferNo - window->firstBufferNo <= 1) TEXT_AND_BREAK("Index series not ready yet")
					else
					{
						if (ImPlot::BeginPlot(plotName.c_str(), "", "", ImVec2(-1, -1), ImPlotFlags_NoTitle, axisFlags, axisFlags))
						{
							plot = ImPlot::GetPlot(plotName.c_str());
							plot->is3d = false;
							if (!window->ShowMultAxes) ImPlot::SetupAxes(KERNEL.usingTime ? "Time" : "Steps", "Index", toAutofitTimeSeries ? ImPlotAxisFlags_AutoFit : 0, 0);
							else
							{
								ImPlot::SetupAxis(ImAxis_X1, KERNEL.usingTime ? "Time" : "Steps", toAutofitTimeSeries ? ImPlotAxisFlags_AutoFit : 0);
								for (int i = 0; i < window->variableCount; i++)
								{
									ImPlot::SetupAxis(ImAxis_Y1 + i, indices[(AnalysisIndex)window->variables[i]].name.c_str(), 0);
								}
							}

							std::vector<numb> Xaxis;
							std::vector<numb> Yaxis;
							numb stepsize = KERNEL.GetStepSize();
							for (int i = 0; i < window->prevbufferNo - window->firstBufferNo; i++)
							{
								Xaxis.push_back(KERNEL.usingTime ? cmp->marshal.variationSize / krnl->VAR_COUNT * stepsize * i : cmp->marshal.variationSize / krnl->VAR_COUNT * i);
							}

							for (int ind = 0; ind < window->variableCount; ind++)
							{
								for (int i = 0; i < window->prevbufferNo - window->firstBufferNo; i++) Yaxis.push_back(window->indSeries[i * window->variableCount + ind]);

								if (window->ShowMultAxes)
								{
									ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1 + ind);
								}

								ImVec4 color;
								if (window->variableCount > 1)
								{
									color = ImPlot::GetColormapColor(ind, window->colormap);
								}

								ImPlot::SetNextLineStyle(window->variableCount > 1 ? color : window->markerColor, window->markerWidth);
								ImPlot::PlotLine(indices[(AnalysisIndex)window->variables[ind]].name.c_str(), Xaxis.data(), Yaxis.data(), window->prevbufferNo - window->firstBufferNo);
								Yaxis.clear();
							}
							
							Xaxis.clear();
							ImPlot::EndPlot();
						}
					}
					
					if (window->whiteBg)
						ImPlot::PopStyleColor();
				}
				break;
			}          

			if (fontNotDefault && window->overrideFontSettings) ImGui::PopFont();
			window->overrideFontSettings = window->overrideFontOnNextFrame;
			ImGui::End();
		}

		if (fontNotDefault) ImGui::PopFont();

		// Rendering
		IMGUI_WORK_END;

		prevVariation = variation;
		prevVariationHires = variationHires;
	}

	saveWindows();
	applicationSettings.Save();

	// Cleanup
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImPlot3D::DestroyContext();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();

	CleanupDeviceD3D();
	::DestroyWindow(guiHwnd);
	::UnregisterClassW(wc.lpszClassName, wc.hInstance);

	terminateComputationBuffers(false);
	terminateUIBuffers();
	unloadPlotWindows();

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

void heatmapRangingSelection(PlotWindow* window, ImPlotPlot* plot, HeatmapSizing* sizing, bool isHires)
{
	Kernel* krnl = !isHires ? &kernelNew : &kernelHiresNew;
	Kernel* krnlComputed = !isHires ? &KERNEL : &kernelHiresComputed;

	int stepX1 = 0;
	int stepY1 = 0;
	int stepX2 = 0;
	int stepY2 = 0;

	if ((!isHires ? window->hmp : window->hireshmp).showActualDiapasons)
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

	int xMaxStep = sizing->hmp->typeX == MDT_Parameter ? krnlComputed->parameters[sizing->hmp->indexX].TrueStepCount() :
		(sizing->hmp->typeX == MDT_Variable ? krnlComputed->variables[sizing->hmp->indexX].TrueStepCount() : 0);
	int yMaxStep = sizing->hmp->typeY == MDT_Parameter ? krnlComputed->parameters[sizing->hmp->indexY].TrueStepCount() :
		(sizing->hmp->typeY == MDT_Variable ? krnlComputed->variables[sizing->hmp->indexY].TrueStepCount() : 0);

	if (sizing->hmp->typeX == MDT_Variable)
	{
		krnl->variables[sizing->hmp->indexX].min = calculateValue(krnlComputed->variables[sizing->hmp->indexX].min, krnlComputed->variables[sizing->hmp->indexX].step, stepX1);
		krnl->variables[sizing->hmp->indexX].max = calculateValue(krnlComputed->variables[sizing->hmp->indexX].min, krnlComputed->variables[sizing->hmp->indexX].step, stepX2);
		krnl->variables[sizing->hmp->indexX].rangingType = RT_Linear;
	}
	else
	{
		krnl->parameters[sizing->hmp->indexX].min = calculateValue(krnlComputed->parameters[sizing->hmp->indexX].min, krnlComputed->parameters[sizing->hmp->indexX].step, stepX1);
		krnl->parameters[sizing->hmp->indexX].max = calculateValue(krnlComputed->parameters[sizing->hmp->indexX].min, krnlComputed->parameters[sizing->hmp->indexX].step, stepX2);
		krnl->parameters[sizing->hmp->indexX].rangingType = RT_Linear;
	}

	if (sizing->hmp->typeY == MDT_Variable)
	{
		krnl->variables[sizing->hmp->indexY].min = calculateValue(krnlComputed->variables[sizing->hmp->indexY].min, krnlComputed->variables[sizing->hmp->indexY].step, stepY1);
		krnl->variables[sizing->hmp->indexY].max = calculateValue(krnlComputed->variables[sizing->hmp->indexY].min, krnlComputed->variables[sizing->hmp->indexY].step, stepY2);
		krnl->variables[sizing->hmp->indexY].rangingType = RT_Linear;
	}
	else
	{
		krnl->parameters[sizing->hmp->indexY].min = calculateValue(krnlComputed->parameters[sizing->hmp->indexY].min, krnlComputed->parameters[sizing->hmp->indexY].step, stepY1);
		krnl->parameters[sizing->hmp->indexY].max = calculateValue(krnlComputed->parameters[sizing->hmp->indexY].min, krnlComputed->parameters[sizing->hmp->indexY].step, stepY2);
		krnl->parameters[sizing->hmp->indexY].rangingType = RT_Linear;
	}

	autoLoadNewParams = false; // Otherwise the map immediately starts drawing the cut region

	if (window->hmp.isHeatmapAutoComputeOn && !isHires)
	{
		if (window->hmp.ignoreLimitsRecalculationOnSelection) 
			for (int w = 0; w < plotWindows.size(); w++)
				plotWindows[w].hmp.ignoreNextLimitsRecalculation = true;
		computeAfterShiftSelect = true;
	}

	if (window->hireshmp.isHeatmapAutoComputeOn)
	{
		if (window->hmp.ignoreLimitsRecalculationOnSelection) 
			for (int w = 0; w < plotWindows.size(); w++)
				plotWindows[w].hmp.ignoreNextLimitsRecalculation = true;
		hiresComputeAfterShiftSelect = true;
	}
}

void hiresShiftClickCompute()
{
	kernelNew.CopyFrom(&kernelHiresComputed);

	for (int v = 0; v < kernelNew.VAR_COUNT; v++)
	{
		kernelNew.variables[v].rangingType = RT_None;
		kernelNew.variables[v].min = valueFromStep(kernelNew.variables[v].min, kernelNew.variables[v].step, attributeValueIndicesHires[v]);
	}

	for (int p = 0; p < kernelNew.PARAM_COUNT; p++)
	{
		if (kernelNew.parameters[p].rangingType != RT_Enum) kernelNew.parameters[p].rangingType = RT_None;
		else
		{
			for (int i = 0; i < MAX_ENUMS; i++) kernelNew.parameters[p].enumEnabled[i] = false;
			int enumChosenInRanging = attributeValueIndicesHires[kernelNew.VAR_COUNT + p];
			for (int i = 0; i < MAX_ENUMS; i++)
			{
				if (kernelHiresComputed.parameters[p].enumEnabled[i])
				{
					if (enumChosenInRanging == 0)
					{
						kernelNew.parameters[p].enumEnabled[i] = true;
						break;
					}
					else
						enumChosenInRanging--;
				}
			}
		}
		kernelNew.parameters[p].min = valueFromStep(kernelNew.parameters[p].min, kernelNew.parameters[p].step, attributeValueIndicesHires[kernelNew.VAR_COUNT + p]);
	}

	autoLoadNewParams = false; // Otherwise the map immediately starts drawing the cut region
	computeAfterShiftSelect = true;
}