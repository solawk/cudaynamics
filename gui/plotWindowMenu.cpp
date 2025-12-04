#include "plotWindowMenu.h"

void plotWindowMenu_File(PlotWindow* window);
void plotWindowMenu_View(PlotWindow* window);
void plotWindowMenu_PhasePlot(PlotWindow* window);
void plotWindowMenu_HeatmapPlot(PlotWindow* window);
void plotWindowMenu_HeatmapColors(PlotWindow* window);
void plotWindowMenu_OrbitPlot(PlotWindow* window);
void plotWindowMenu_MetricPlot(PlotWindow* window);
void plotWindowMenu_SeriesPlot(PlotWindow* window);
void plotWindowMenu_IndSeriesPlot(PlotWindow* window);

extern bool enabledParticles;
extern bool autofitHeatmap;
extern PlotWindow* colorsLUTfrom;
extern int paintLUTsize;
extern AnalysisIndex hiresIndex;
extern Kernel kernelNew, kernelHiresNew, kernelHiresComputed;

void plotWindowMenu(PlotWindow* window)
{
	if (ImGui::BeginMenuBar())
	{
		plotWindowMenu_File(window);
		plotWindowMenu_View(window);
		if (window->type == Phase || window->type == Phase2D) plotWindowMenu_PhasePlot(window);
		if (window->type == Heatmap || window->type == MCHeatmap) plotWindowMenu_HeatmapPlot(window);
		if (window->type == Heatmap) plotWindowMenu_HeatmapColors(window);
		if (window->type == Heatmap)
		{
			ImGui::Text("   ");
			ImGui::SameLine();

			bool isHires = window->isTheHiresWindow(hiresIndex);
			if (ImGui::Checkbox(("Hi-Res Mode##" + window->name + "_hirescheckbox").c_str(), &isHires))
			{
				if (window->isTheHiresWindow(hiresIndex))
				{
					hiresIndex = IND_NONE;
				}
				else
				{
					if (hiresIndex == IND_NONE) kernelHiresNew.CopyFrom(&kernelNew);
					hiresIndex = (AnalysisIndex)window->variables[0];
					//kernelHiresNew.CopyFrom(&kernelNew);
				}
			}
		}
		if (window->type == Orbit) plotWindowMenu_OrbitPlot(window);
		if (window->type == Metric) plotWindowMenu_MetricPlot(window);
		if (window->type == VarSeries) plotWindowMenu_SeriesPlot(window);
		if (window->type == IndSeries) plotWindowMenu_IndSeriesPlot(window);
		ImGui::EndMenuBar();
	}
}

void plotWindowMenu_File(PlotWindow* window)
{
	if (ImGui::BeginMenu("File"))
	{
		if (ImGui::MenuItem("Export to .csv", nullptr, false,
			(window->type == Heatmap) || (window->type == VarSeries)))
		{
			std::string savedPath;
			bool attempted = false; 

			switch (window->type)
			{
				// === HEATMAP (LLE / MAX / ...) ===
			case Heatmap:
			{
				const bool isHires = window->isTheHiresWindow(hiresIndex);
				const HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;
				Kernel* krnl = isHires ? &kernelHiresComputed : &(KERNEL);

				const int valuesCount = isHires ? window->hireshmp.lastBufferSize
					: window->hmp.lastBufferSize;
				if (!heatmap->values.valueBuffer) {
					MessageBoxA(NULL, "Export failed: Heatmap buffer is null.", "Export", MB_OK | MB_ICONERROR);
					break;
				}
				if (valuesCount <= 0) {
					MessageBoxA(NULL, "Export failed: Heatmap buffer is empty.", "Export", MB_OK | MB_ICONERROR);
					break;
				}
				if (window->variables.empty()) {
					MessageBoxA(NULL, "Export failed: no map selected (window->variables is empty).", "Export", MB_OK | MB_ICONERROR);
					break;
				}
				const AnalysisIndex mapIdx = (AnalysisIndex)window->variables[0];
				if (mapIdx < 0 || mapIdx >= indices.size()) {
					MessageBoxA(NULL, "Export failed: map index out of range.", "Export", MB_OK | MB_ICONERROR);
					break;
				}

				HeatmapSizing sizing;
				sizing.loadPointers(krnl, const_cast<HeatmapProperties*>(heatmap));
				sizing.initValues();

				const std::string mapName = indices[mapIdx].name;
				savedPath = exportHeatmapCSV(mapName, sizing, heatmap);
				attempted = true;
				break;
			}

			// === TIME SERIES ===
			case VarSeries:
			{
				savedPath = exportTimeSeriesCSV(window);
				attempted = true;
				break;
			}

			default:
			{
				MessageBoxA(NULL, "Export not supported for this plot type.", "Export", MB_OK | MB_ICONWARNING);
				break;
			}
			}

			// ≈диный блок уведомлени¤ о результате 
			if (attempted) {
				if (!savedPath.empty()) {
					std::string msg = "CSV saved to:\n" + savedPath;
					MessageBoxA(NULL, msg.c_str(), "Export", MB_OK | MB_ICONINFORMATION);
				}
				else {
					MessageBoxA(NULL, "Export failed (empty data or I/O error).", "Export", MB_OK | MB_ICONERROR);
				}
			}
		}

		ImGui::EndMenu();
	}
}

void plotWindowMenu_View(PlotWindow* window)
{
	if (ImGui::BeginMenu("View"))
	{
		
		std::string windowName = window->name + std::to_string(window->id);

		if (window->type == Phase)
		{
			bool tempSettingsEnabled = window->settingsListEnabled; if (ImGui::Checkbox(("##" + windowName + "setList").c_str(), &tempSettingsEnabled)) window->settingsListEnabled = !window->settingsListEnabled;
			ImGui::SameLine(); ImGui::Text("View settings");
		}
		
		bool tempOverrideFont = window->overrideFontSettings;
		if (ImGui::Checkbox(("Override font settings##" + window->name).c_str(), &tempOverrideFont))
		{
			if (!window->overrideFontSettings)	window->overrideFontOnNextFrame = true;
			else								window->overrideFontOnNextFrame = false;
		}
		if (window->overrideFontSettings) FontMenu(window);

		ImGui::EndMenu();
	}
}

void plotWindowMenu_CommonPlot(PlotWindow* window, std::string windowName)
{
	bool tempWhiteBg = window->whiteBg; if (ImGui::Checkbox(("##" + windowName + "whiteBG").c_str(), &tempWhiteBg)) window->whiteBg = !window->whiteBg;
	ImGui::SameLine(); ImGui::Text("White background");

}

void plotWindowMenu_PhasePlot(PlotWindow* window)
{
	if (ImGui::BeginMenu("Plot"))
	{
		std::string windowName = window->name + std::to_string(window->id);
		plotWindowMenu_CommonPlot(window, windowName);

		if (enabledParticles)
		{
			ImGui::ColorEdit4(("##" + windowName + "_particleColor").c_str(), (float*)(&(window->markerColor)));		ImGui::SameLine(); ImGui::Text("Particle color");
			ImGui::DragFloat(("##" + windowName + "_particleSize").c_str(), &(window->markerWidth), 0.1f);				ImGui::SameLine(); ImGui::Text("Particle size");
			ImGui::DragFloat(("##" + windowName + "_particleOutlineSize").c_str(), &(window->markerOutlineWidth), 0.1f);	ImGui::SameLine(); ImGui::Text("Particle outline size");
			if (window->markerWidth < 0.0f) window->markerWidth = 0.0f;
		}
		else
		{
			ImGui::ColorEdit4(("##" + windowName + "_lineColor").c_str(), (float*)(&(window->plotColor)));		ImGui::SameLine(); ImGui::Text("Line color");
		}

		bool tempDAT = window->drawAllTrajectories; if (ImGui::Checkbox(("##" + windowName + "allTraj").c_str(), &tempDAT)) window->drawAllTrajectories = !window->drawAllTrajectories;
		ImGui::SameLine(); ImGui::Text("Draw all trajectories (intensive!)");

		if (window->type == Phase)
		{
			bool tempIsI3d = window->isImplot3d; if (ImGui::Checkbox(("##" + windowName + "isI3D").c_str(), &tempIsI3d)) window->isImplot3d = !window->isImplot3d;
			ImGui::SameLine(); ImGui::Text("Use ImPlot3D");

			//bool tempSettingsEnabled = window->settingsListEnabled; if (ImGui::Checkbox(("##" + windowName + "setList").c_str(), &tempSettingsEnabled)) window->settingsListEnabled = !window->settingsListEnabled;
			//ImGui::SameLine(); ImGui::Text("View settings");
		}

		ImGui::EndMenu();
	}
}

void plotWindowMenu_OrbitPlot(PlotWindow* window) {
	if (ImGui::BeginMenu("Plot"))
	{
		std::string windowName = window->name + std::to_string(window->id);
		
		ImGui::SeparatorText("Plot Type");
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.58f);
		std::string orbitplottypes[] = { "Peaks / Parameter", "Intervals / Parameter", "Peaks / Intervals" , "Peaks / Intervals / Parameter"};
		if (ImGui::BeginCombo(("##" + windowName + "_OrbitPlotType").c_str(), (orbitplottypes[window->OrbitType]).c_str(), 0))
		{
			for (int t = 0; t < OrbitPlotType_COUNT; t++)
			{
				bool isSelected = window->OrbitType == t;
				ImGuiSelectableFlags selectableFlags = 0;
				if (ImGui::Selectable(orbitplottypes[t].c_str(), isSelected, selectableFlags)) window->OrbitType = (OrbitPlotType)t;
			}

			ImGui::EndCombo();
		}
		ImGui::SameLine();
		ImGui::Text("Orbit plot type");

		ImGui::SeparatorText("Point");
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.485f);
		std::string orbitdottypes[] = { "Circle", "Square", "Diamond", "Up Triangle", "Down Triangle", "Left Triangle", "Right Triangle", "Cross", "Plus", "Asterisk"};
		if (ImGui::BeginCombo(("##" + windowName + "_OrbitDotShape").c_str(), (orbitdottypes[window->markerShape]).c_str(), 0))
		{
			for (int t = 0; t < ImPlotMarker_COUNT; t++)
			{
				bool isSelected = window->OrbitType == t;
				ImGuiSelectableFlags selectableFlags = 0;
				if (ImGui::Selectable(orbitdottypes[t].c_str(), isSelected, selectableFlags)) window->markerShape = (ImPlotMarker)t;
			}

			ImGui::EndCombo();
		}
		ImGui::SameLine(); ImGui::Text("Point shape");
		ImGui::ColorEdit4(("##" + windowName + "_dotColor").c_str(), (float*)(&(window->plotColor)));		ImGui::SameLine(); ImGui::Text("Point color");
		ImGui::DragFloat("Point size", &window->OrbitPointSize, 0.1f, 0.5f, 4.0f,"%.1f");

		ImGui::SeparatorText("Marker");
		ImGui::Checkbox("Show parameter marker", &window->ShowOrbitParLines);
		ImGui::ColorEdit4(("##" + windowName + "_markerColor").c_str(), (float*)(&(window->OrbitMarkerColor)));		ImGui::SameLine(); ImGui::Text("Marker color");
		ImGui::DragFloat("Marker width", &window->OrbitMarkerWidth, 0.1f, 0.5f, 4.0f, "%.1f");

		ImGui::SeparatorText("Other settings");
		ImGui::Checkbox("Invert axes", &window->OrbitInvertedAxes);
		plotWindowMenu_CommonPlot(window, windowName);

		

		ImGui::SeparatorText("Auto-compute");
		ImGui::Checkbox("Auto - compute on Shift + RMB", &window->isAutoComputeOn);

		ImGui::EndMenu();
	}
}

void plotWindowMenu_HeatmapPlot(PlotWindow* window)
{
	if (ImGui::BeginMenu("Plot"))
	{
		bool isHires = window->isTheHiresWindow(hiresIndex);
		HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;

		std::string windowName = window->name + std::to_string(window->id);

		ImGui::SeparatorText("Plot View");

		plotWindowMenu_CommonPlot(window, windowName);

		bool tempIsDelta = window->isDelta; if (ImGui::Checkbox(("##" + windowName + "isDelta").c_str(), &tempIsDelta))
		{
			window->isDelta = !window->isDelta;
			heatmap->areValuesDirty = true;
		}
		ImGui::SameLine(); ImGui::Text("Show delta");

		std::string valueDisplayStrings[] = { "No display value", "Value of selected variation", "Split", "Value under mouse" };
		std::string valueDisplayTooltips[] = {
			"Don't show any heatmap value in the corner and on the color scale",
			"Show heatmap value of selected variation in the corner and on the scale",
			"Show value of the variation on the scale and value under mouse cursor in the corner",
			"Show heatmap value under mouse cursor in the corner and on the scale"
		};

		//std::string valueDisplayStrings[] = { "No value in the corner", "Value of selected variation", "Value under mouse" };
		if (ImGui::BeginCombo(("##" + windowName + "valueDisplay").c_str(), valueDisplayStrings[heatmap->valueDisplay].c_str()))
		{
			for (int i = 0; i < 4; i++) 
			{
				if (ImGui::Selectable(valueDisplayStrings[i].c_str(), (int)heatmap->valueDisplay == i)) heatmap->valueDisplay = (ValueDisplayMode)i;
				TOOLTIP(valueDisplayTooltips[i].c_str());
			}
			ImGui::EndCombo();
		}

		bool tempShowHeatmapValues = heatmap->showHeatmapValues; if (ImGui::Checkbox(("##" + windowName + "showHeatmapValues").c_str(), &tempShowHeatmapValues)) heatmap->showHeatmapValues = !heatmap->showHeatmapValues;
		ImGui::SameLine(); ImGui::Text("Show values");

		bool tempShowDragLines = heatmap->showDragLines; if (ImGui::Checkbox(("##" + windowName + "showDragLines").c_str(), &tempShowDragLines)) heatmap->showDragLines = !heatmap->showDragLines;
		ImGui::SameLine(); ImGui::Text("Show crosshair lines");

		ImGui::ColorEdit4(("##" + windowName + "_markerColor").c_str(), (float*)(&(window->markerColor)));		ImGui::SameLine(); ImGui::Text("Marker color");
		ImGui::DragFloat("Marker width", &window->markerWidth, 0.1f, 0.5f, 4.0f, "%.1f");

		bool tempShowLegend = heatmap->showLegend; if (ImGui::Checkbox(("##" + windowName + "showLegend").c_str(), &tempShowLegend)) heatmap->showLegend = !heatmap->showLegend;
		ImGui::SameLine(); ImGui::Text("Show colormap");

		std::string diapasonsStrings[] = { "Values", "Steps" };
		bool tempShowActualDiapasons = heatmap->showActualDiapasons;
		if (ImGui::BeginCombo(("##" + windowName + "diapasons").c_str(), (heatmap->showActualDiapasons ? diapasonsStrings[0] : diapasonsStrings[1]).c_str()))
		{
			if (ImGui::Selectable(diapasonsStrings[0].c_str(), heatmap->showActualDiapasons)) heatmap->showActualDiapasons = true;
			if (ImGui::Selectable(diapasonsStrings[1].c_str(), !heatmap->showActualDiapasons)) heatmap->showActualDiapasons = false;
			ImGui::EndCombo();
		}

		ImGui::SeparatorText("Auto-compute");

		bool tempHeatmapAutoCompute = heatmap->isHeatmapAutoComputeOn; if (ImGui::Checkbox(("##" + windowName + "heatmapAutoCompute").c_str(), &tempHeatmapAutoCompute)) heatmap->isHeatmapAutoComputeOn = !heatmap->isHeatmapAutoComputeOn;
		ImGui::SameLine(); ImGui::Text("Auto-compute on Shift+RMB");

		bool tempIgnoreLimitsRecalc = heatmap->ignoreLimitsRecalculationOnSelection; if (ImGui::Checkbox(("##" + windowName + "heatmapignoreLimitsRecalc").c_str(), &tempIgnoreLimitsRecalc)) heatmap->ignoreLimitsRecalculationOnSelection = !heatmap->ignoreLimitsRecalculationOnSelection;
		ImGui::SameLine(); ImGui::Text("Auto-compute does not update limits");

		ImGui::EndMenu();
	}
}

void plotWindowMenu_HeatmapColors(PlotWindow* window)
{
	if (ImGui::BeginMenu("Colors"))
	{
		std::string windowName = window->name + std::to_string(window->id);
		bool isHires = window->isTheHiresWindow(hiresIndex);
		HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;

		std::string colormapStrings[] = { "Deep", "Dark", "Pastel", "Paired", "Viridis", "Plasma", "Hot", "Cool", "Pink", "Jet", "Twilight", "RdBu", "BrBG", "PiYG", "Spectral", "Greys" };
		ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
		if (ImGui::BeginCombo(("##" + windowName + "colormap").c_str(), (colormapStrings[heatmap->colormap]).c_str()))
		{
			for (int i = 0; i < 16; i++)
				if (ImGui::Selectable(colormapStrings[i].c_str(), heatmap->colormap == i))
				{
					heatmap->colormap = i;
					heatmap->isHeatmapDirty = true;
				}
			
			ImGui::EndCombo();
		}
		ImGui::PopItemFlag();

		if (colorsLUTfrom != window)
		{
			if (ImGui::Button("Use heatmap for painting"))
			{
				colorsLUTfrom = window;
			}
		}
		else
		{
			if (ImGui::Button("Stop using heatmap for painting"))
			{
				colorsLUTfrom = nullptr;
			}
		}

		int tempLS = paintLUTsize;
		ImGui::DragInt(("##" + windowName + "_paintLUT").c_str(), &(paintLUTsize));	ImGui::SameLine(); ImGui::Text("Colors");
		if (paintLUTsize < 2) paintLUTsize = 2;
		if (tempLS != paintLUTsize) window->hmp.isHeatmapDirty = true;

		ImGui::EndMenu();
	}
}

void plotWindowMenu_MetricPlot(PlotWindow*window) {
	if (ImGui::BeginMenu("Plot")) {
		std::string windowName = window->name + std::to_string(window->id);
		ImGui::SeparatorText("Plot View");
		plotWindowMenu_CommonPlot(window, windowName);

		if (window->variableCount==1) {
			ImGui::ColorEdit4(("##" + windowName + "_lineColor").c_str(), (float*)(&(window->markerColor)));		ImGui::SameLine(); ImGui::Text("Line color");
		}
		else {
			std::string colormapStrings[] = {"Deep", "Dark", "Pastel", "Paired", "Viridis", "Plasma", "Hot", "Cool", "Pink", "Jet", "Twilight", "RdBu", "BrBG", "PiYG", "Spectral", "Greys"};
			ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
			if (ImGui::BeginCombo(("##" + windowName + "colormap").c_str(), (colormapStrings[window->colormap]).c_str()))
			{
				for (int i = 0; i < 16; i++)
					if (ImGui::Selectable(colormapStrings[i].c_str(), window->colormap == i))
					{
						window->colormap = i;
					}

				ImGui::EndCombo();
			}
			ImGui::PopItemFlag();
			
		}
		ImGui::DragFloat("Line width", &window->markerWidth, 0.1f, 0.5f, 4.0f, "%.1f");

		ImGui::Checkbox("Show parameter marker", &window->ShowOrbitParLines);
		ImGui::ColorEdit4(("##" + windowName + "_markerColor").c_str(), (float*)(&(window->OrbitMarkerColor)));		ImGui::SameLine(); ImGui::Text("Marker color");
		ImGui::DragFloat("Marker width", &window->OrbitMarkerWidth, 0.1f, 0.5f, 4.0f, "%.1f");

		ImGui::Checkbox("Show multiple axes", &window->ShowMultAxes);

		ImGui::SeparatorText("Auto-compute");
		ImGui::Checkbox("Auto - compute on Shift + RMB", &window->isAutoComputeOn);

		ImGui::EndMenu();
	}
}

void plotWindowMenu_SeriesPlot(PlotWindow* window) {
	if (ImGui::BeginMenu("Plot")) {

		ImGui::SeparatorText("Plot View");
		std::string windowName = window->name + std::to_string(window->id);
		plotWindowMenu_CommonPlot(window, windowName);

		if (window->variableCount == 1) {
			ImGui::ColorEdit4(("##" + windowName + "_lineColor").c_str(), (float*)(&(window->markerColor)));		ImGui::SameLine(); ImGui::Text("Line color");
		}
		else {
			std::string colormapStrings[] = { "Deep", "Dark", "Pastel", "Paired", "Viridis", "Plasma", "Hot", "Cool", "Pink", "Jet", "Twilight", "RdBu", "BrBG", "PiYG", "Spectral", "Greys" };
			ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
			if (ImGui::BeginCombo(("##" + windowName + "colormap").c_str(), (colormapStrings[window->colormap]).c_str()))
			{
				for (int i = 0; i < 16; i++)
					if (ImGui::Selectable(colormapStrings[i].c_str(), window->colormap == i))
					{
						window->colormap = i;
					}

				ImGui::EndCombo();
			}
			ImGui::PopItemFlag();

		}
		ImGui::DragFloat("Line width", &window->markerWidth, 0.1f, 0.5f, 4.0f, "%.1f");

		ImGui::Checkbox("Show multiple axes", &window->ShowMultAxes);



		ImGui::EndMenu();
	}
}

void plotWindowMenu_IndSeriesPlot(PlotWindow* window) {
	if (ImGui::BeginMenu("Plot")) {

		ImGui::SeparatorText("Plot View");
		std::string windowName = window->name + std::to_string(window->id);
		plotWindowMenu_CommonPlot(window, windowName);

		if (window->variableCount == 1) {
			ImGui::ColorEdit4(("##" + windowName + "_lineColor").c_str(), (float*)(&(window->markerColor)));		ImGui::SameLine(); ImGui::Text("Line color");
		}
		else {
			std::string colormapStrings[] = { "Deep", "Dark", "Pastel", "Paired", "Viridis", "Plasma", "Hot", "Cool", "Pink", "Jet", "Twilight", "RdBu", "BrBG", "PiYG", "Spectral", "Greys" };
			ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
			if (ImGui::BeginCombo(("##" + windowName + "colormap").c_str(), (colormapStrings[window->colormap]).c_str()))
			{
				for (int i = 0; i < 16; i++)
					if (ImGui::Selectable(colormapStrings[i].c_str(), window->colormap == i))
					{
						window->colormap = i;
					}

				ImGui::EndCombo();
			}
			ImGui::PopItemFlag();

		}
		ImGui::DragFloat("Line width", &window->markerWidth, 0.1f, 0.5f, 4.0f, "%.1f");

		ImGui::Checkbox("Show multiple axes", &window->ShowMultAxes);



		ImGui::EndMenu();
	}
}