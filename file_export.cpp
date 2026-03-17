#include "file_export.h"

extern Computation computations[2];
extern int playedBufferIndex;
extern int computedSteps;
extern int bufferNo;
extern std::vector<int> attributeValueIndices;
extern void steps2Variation(uint64_t* variationIndex, int* avi, Kernel* kernel);
extern std::map<AnalysisIndex, Index> indices;
extern bool enabledParticles;
extern int particleStep;

namespace {
    std::string make_safe(std::string s) {
        for (char& c : s) if (c == ' ') c = '_';
        return s.empty() ? std::string("Untitled") : s;
    }

    std::string timestamp_yyyy_mm_dd_hh_mm_ss() {
        char ts[32];
        std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_s(&tm, &t);
        std::snprintf(ts, sizeof(ts), "%04d-%02d-%02d_%02d-%02d-%02d",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec);
        return ts;
    }

    // Creates export/ directory and builds path. extra may be empty.
    std::string build_export_path(std::string_view systemName,
        std::string_view artifactName,   // example: "timeSeries" or map name
        std::string_view extra,          // example: suffix or left empty
        std::string_view ext = ".csv") {
        std::filesystem::create_directories("export");
        std::string base = "export/";
        base += std::string(systemName);
        if (!artifactName.empty()) {
            base += "_";
            base += std::string(artifactName);
        }
        if (!extra.empty()) {
            base += "_";
            base += std::string(extra);
        }
        base += "_";
        base += timestamp_yyyy_mm_dd_hh_mm_ss();
        base += std::string(ext);
        return base;
    }

    std::ofstream open_csv(const std::string& path) {
        std::ofstream f(path);
        if (f.is_open()) {
            f.imbue(std::locale::classic());
            f << std::fixed << std::setprecision(9);
        }
        return f;
    }

    // axis names for heatmap
    std::string axis_name(const Kernel& k, MapDimensionType t, int idx) {
        switch (t) {
        case MapDimensionType::MDT_Variable:
            return (idx >= 0 && idx < k.VAR_COUNT) ? k.variables[idx].name : "var";
        case MapDimensionType::MDT_Parameter:
            return (idx >= 0 && idx < k.PARAM_COUNT) ? k.parameters[idx].name : "par";
        case MapDimensionType::MDT_Step:
            return "step";
        default:
            return "";
        }
    }

    // Unification of system name
    std::string safe_system_name(const Kernel& k) {
        return make_safe(k.name.empty() ? std::string("System") : k.name);
    }
} // namespace

// ======================================================
//                SIMPLE FILE EXPORT
// ======================================================
void exportToFile(std::string name, numb* values, int count)
{
    if (!values || count <= 0) return;
    std::ofstream out = open_csv(name);
    if (!out.is_open()) return;

    for (int i = 0; i < count; ++i)
        out << values[i] << '\n';
}

// ======================================================
//                HEATMAP EXPORT 
// ======================================================
std::string exportHeatmapCSV(const std::string& mapName,
    const HeatmapSizing& s,
    const HeatmapProperties* h)
{
    if (!h || !s.kernel || !s.hmp) return {};
    const Kernel& k = *s.kernel;
    const numb* vals = h->values.valueBuffer;
    if (!vals || s.xSize <= 0 || s.ySize <= 0) return {};

    const std::string systemName = safe_system_name(k);
    const std::string mapSafe = make_safe(mapName.empty() ? std::string("Map") : mapName);

    const std::string nameX = [&] {
        auto n = axis_name(k, h->typeX, h->indexX);
        return n.empty() ? std::string("X") : n;
        }();
    const std::string nameY = [&] {
        auto n = axis_name(k, h->typeY, h->indexY);
        return n.empty() ? std::string("Y") : n;
        }();

    const std::string path = build_export_path(systemName, mapSafe, /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    // Header
    f << nameY << "\\" << nameX;
    for (int x = 0; x < s.xSize; ++x)
        f << ',' << (s.minX + s.stepX * (numb)x);
    f << '\n';

    // Data
    for (int y = 0; y < s.ySize; ++y) {
        f << (s.minY + s.stepY * (numb)y);
        const size_t base = (size_t)y * (size_t)s.xSize;
        for (int x = 0; x < s.xSize; ++x)
            f << ',' << vals[base + x];
        f << '\n';
    }

    return path;
}

// ======================================================
//                TIME SERIES EXPORT
// ======================================================
std::string exportTimeSeriesCSV(const PlotWindow* window)
{
    if (!window) return {};

    // active computation
    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
        return {};

    // Checking step and variable
    const int steps = computedSteps;
    if (steps <= 0 || KERNEL.VAR_COUNT <= 0)
        return {};

    // variation index
    uint64_t variation = 0;
    if (!attributeValueIndices.empty())
        steps2Variation(&variation, attributeValueIndices.data(), &KERNEL);

    const uint64_t varStride = comp.marshal.variationSize;
    const numb* base = comp.marshal.trajectory + varStride * (uint64_t)variation;

    // file name
    const std::string systemName = safe_system_name(KERNEL);
    const std::string path = build_export_path(systemName, "timeSeries", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    // X axis
    const bool useTime = KERNEL.usingTime;
    const numb stepSize = KERNEL.GetStepSize();
    const double start = (double)(bufferNo * KERNEL.steps + KERNEL.transientSteps) *
        (useTime ? (double)stepSize : 1.0);

    // === Title ===
    f << (useTime ? "time" : "step");
    for (int v : window->variables) {
        if (v >= 0 && v < KERNEL.VAR_COUNT)
            f << ',' << KERNEL.variables[v].name;
        else
            f << ',';
    }
    f << '\n';

    // === Data ===
    const int rows = steps + 1;
    for (int i = 0; i < rows; ++i) {
        const double x = start + (useTime ? i * (double)stepSize : i);
        f << x;

        const numb* row = base + (uint64_t)i * (uint64_t)KERNEL.VAR_COUNT;
        for (int v : window->variables) {
            if (v < 0 || v >= KERNEL.VAR_COUNT) { f << ','; continue; }
            f << ',' << row[v];
        }
        f << '\n';
    }

    return path;
}

// ======================================================
//                DECAY EXPORT
// ======================================================
std::string exportDecayCSV(const PlotWindow* window)
{
    if (!window) return {};

    // Active computation
    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
        return {};

    // Checks steps and variable
    const int steps = computedSteps;
    if (steps <= 0 || KERNEL.VAR_COUNT <= 0)
        return {};

    // file name
    const std::string systemName = safe_system_name(KERNEL);
    const std::string path = build_export_path(systemName, "decay", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    // X axis
    const bool useTime = KERNEL.usingTime;

    // === Заголовок ===
    f << (useTime ? "time" : "step");
    Index* index = &(indices[(AnalysisIndex)window->variables[0]]);
    int thresholdCount = (int)index->decay.thresholds.size();
    for (float threshold : index->decay.thresholds) 
    {
        f << ',' << threshold;
    }
    f << ",total" << '\n';

    // === Данные ===
    const int rows = (int)window->decay.buffer[0].size();
    for (int i = 0; i < rows; ++i) 
    {
        const double x = window->decay.buffer[0][i];
        f << x;

        for (int t = 0; t < thresholdCount; t++)
        {
            f << ',' << window->decay.alive[t][i];
        }

        f << ',' << window->decay.total[0][i] << '\n';
    }

    return path;
}

// ======================================================
//                ORBIT EXPORT
// ======================================================
std::string exportOrbitCSV(const PlotWindow* window)
{
    if (!window) return {};

    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
        return {};

    const int steps = computedSteps;
    if (steps <= 0 || KERNEL.VAR_COUNT <= 0)
        return {};

    Attribute* axis = &(KERNEL.parameters[window->orbit.xIndex]);

    const std::string systemName = safe_system_name(KERNEL);

    // Normal orbit
    std::string path = build_export_path(systemName, "orbit", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    switch (window->orbit.type)
    {
    case OPT_Peak_Bifurcation:
        f << axis->name << ",Peaks" << '\n';
        for (int i = 0; i < window->orbit.bifDotAmount; ++i)
        {
            f << window->orbit.bifParamIndices[i] << "," << window->orbit.bifAmps[i] << '\n';
        }
        break;
    case OPT_Interval_Bifurcation:
        f << axis->name << ",Intervals" << '\n';
        for (int i = 0; i < window->orbit.bifDotAmount; ++i)
        {
            f << window->orbit.bifParamIndices[i] << "," << window->orbit.bifIntervals[i] << '\n';
        }
        break;
    case OPT_Bifurcation_3D:
        f << axis->name << ",Peaks,Intervals" << '\n';
        for (int i = 0; i < window->orbit.bifDotAmount; ++i)
        {
            f << window->orbit.bifParamIndices[i] << "," << window->orbit.bifAmps[i] << "," << window->orbit.bifIntervals[i] << '\n';
        }
        break;
    case OPT_Selected_Var_Section:
        f << axis->name << ",Peaks,Intervals" << '\n';
        for (int i = 0; i < window->orbit.peakCount - 1; ++i)
        {
            f << i << ',' << window->orbit.peakAmplitudes[i] << "," << window->orbit.peakIntervals[i] << '\n';
        }
        break;
    }

    if (window->orbit.drawingContinuation)
    {
        // Forward and backward continuations

        path = build_export_path(systemName, "orbit-forward", /*extra*/"", ".csv");
        f = open_csv(path);
        if (!f.is_open()) return {};

        switch (window->orbit.type)
        {
        case OPT_Peak_Bifurcation:
            f << axis->name << ",Peaks" << '\n';
            for (int i = 0; i < window->orbit.bifDotAmountForward; ++i)
            {
                f << window->orbit.continuationParamIndicesForward[i] << "," << window->orbit.continuationAmpsForward[i] << '\n';
            }
            break;
        case OPT_Interval_Bifurcation:
            f << axis->name << ",Intervals" << '\n';
            for (int i = 0; i < window->orbit.bifDotAmountForward; ++i)
            {
                f << window->orbit.continuationParamIndicesForward[i] << "," << window->orbit.continuationIntervalsForward[i] << '\n';
            }
            break;
        case OPT_Bifurcation_3D:
            f << axis->name << ",Peaks,Intervals" << '\n';
            for (int i = 0; i < window->orbit.bifDotAmountForward; ++i)
            {
                f << window->orbit.continuationParamIndicesForward[i] << "," << window->orbit.continuationAmpsForward[i] << "," << window->orbit.continuationIntervalsForward[i] << '\n';
            }
            break;
        case OPT_Selected_Var_Section:
            f << axis->name << ",Peaks,Intervals" << '\n';
            const numb* amps = window->orbit.SliceForwardPeak.data();
            const numb* ints = window->orbit.SliceForwardInt.data();
            for (int i = 0; i < window->orbit.forNum - 1; ++i)
            {
                f << i << ',' << amps[i] << "," << ints[i] << '\n';
            }
            break;
        }

        path = build_export_path(systemName, "orbit-backward", /*extra*/"", ".csv");
        f = open_csv(path);
        if (!f.is_open()) return {};

        switch (window->orbit.type)
        {
        case OPT_Peak_Bifurcation:
            f << axis->name << ",Peaks" << '\n';
            for (int i = 0; i < window->orbit.bifDotAmountBack; ++i)
            {
                f << window->orbit.continuationParamIndicesBack[i] << "," << window->orbit.continuationAmpsBack[i] << '\n';
            }
            break;
        case OPT_Interval_Bifurcation:
            f << axis->name << ",Intervals" << '\n';
            for (int i = 0; i < window->orbit.bifDotAmountBack; ++i)
            {
                f << window->orbit.continuationParamIndicesBack[i] << "," << window->orbit.continuationIntervalsBack[i] << '\n';
            }
            break;
        case OPT_Bifurcation_3D:
            f << axis->name << ",Peaks,Intervals" << '\n';
            for (int i = 0; i < window->orbit.bifDotAmountBack; ++i)
            {
                f << window->orbit.continuationParamIndicesBack[i] << "," << window->orbit.continuationAmpsBack[i] << "," << window->orbit.continuationIntervalsBack[i] << '\n';
            }
            break;
        case OPT_Selected_Var_Section:
            f << axis->name << ",Peaks,Intervals" << '\n';
            const numb* amps = window->orbit.SliceBackwardPeak.data();
            const numb* ints = window->orbit.SliceBackwardInt.data();
            for (int i = 0; i < window->orbit.backNum - 1; ++i)
            {
                f << i << ',' << amps[i] << "," << ints[i] << '\n';
            }
            break;
        }
    }

    return path;
}

// ======================================================
//          INDICES SERIES / DIAGRAM EXPORT (FIXED)
// ======================================================
std::string exportIndicesSeriesCSV(const PlotWindow* window)
{
    if (window->type != IndSeries) {
        printf("[export] ERROR: window type is not IndSeries\n");
        return {};
    }

    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory) {
        printf("[export] ERROR: computation not ready\n");
        return {};
    }

    if (window->indSeries.empty()) {
        printf("[export] WARNING: indSeries is empty, nothing to export\n");
        return {};
    }

    const std::vector<int>& selectedIndices = window->variables;
    const size_t numIndices = selectedIndices.size();
    if (numIndices == 0) {
        printf("[export] ERROR: no indices selected\n");
        return {};
    }

    size_t totalValues = window->indSeries.size();
    size_t numPoints = totalValues / numIndices;
    if (numPoints * numIndices != totalValues) {
        printf("[export] ERROR: indSeries size (%zu) is not multiple of numIndices (%zu)\n", totalValues, numIndices);
        return {};
    }

    const bool useTime = KERNEL.usingTime;
    std::string xAxisName = useTime ? "time" : "steps";

    unsigned long long stepsPerVariation = comp.marshal.variationSize / KERNEL.VAR_COUNT;
    const numb stepSize = KERNEL.GetStepSize();

    const std::string systemName = safe_system_name(KERNEL);
    const std::string path = build_export_path(systemName, "indices", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) {
        printf("[export] ERROR: cannot open file %s\n", path.c_str());
        return {};
    }

    f << xAxisName;
    for (int idx : selectedIndices) {
        auto it = indices.find((AnalysisIndex)idx);
        if (it != indices.end())
            f << ',' << it->second.name;
        else
            f << ",unknown";
    }
    f << '\n';

    for (size_t point = 0; point < numPoints; ++point) {
        double xValue;
        if (useTime) {
            xValue = (double)(point * stepsPerVariation * stepSize);
        }
        else {
            xValue = (double)(point * stepsPerVariation);
        }

        f << xValue;
        for (size_t idx = 0; idx < numIndices; ++idx) {
            size_t dataIndex = point * numIndices + idx;
            f << ',' << window->indSeries[dataIndex];
        }
        f << '\n';
    }

    return path;
}

// ======================================================
//                2D PHASE DIAGRAM EXPORT
// ======================================================
std::string exportPhase2DCSV(const PlotWindow* window)
{
    if (!window)
    {
        printf("[export] ERROR: window is null\n");
        return {};
    }

    if (window->type != Phase2D)
    {
        printf("[export] ERROR: window is not Phase2D\n");
        return {};
    }

    if (window->variables.size() < 2)
    {
        printf("[export] ERROR: Not enough variables for 2D phase diagram (need at least 2)\n");
        return {};
    }

    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
    {
        printf("[export] ERROR: Computation not ready\n");
        return {};
    }

    const int steps = computedSteps;
    if (steps <= 0)
    {
        printf("[export] ERROR: No steps computed\n");
        return {};
    }

    int xIdx = window->variables[0];
    int yIdx = window->variables[1];

    if (xIdx < 0 || xIdx >= KERNEL.VAR_COUNT || yIdx < 0 || yIdx >= KERNEL.VAR_COUNT)
    {
        printf("[export] ERROR: Invalid variable indices (x=%d, y=%d, VAR_COUNT=%d)\n",
            xIdx, yIdx, KERNEL.VAR_COUNT);
        return {};
    }

    extern bool enabledParticles;

    bool isTrajectoryMode = (!enabledParticles);

    const std::string systemName = safe_system_name(KERNEL);
    std::string mode = isTrajectoryMode ? "trajectory" : "particles";

    std::string path = build_export_path(systemName, "phase2d_" + mode, "", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open())
    {
        printf("[export] ERROR: Cannot open file %s\n", path.c_str());
        return {};
    }

    f << KERNEL.variables[xIdx].name << "," << KERNEL.variables[yIdx].name;

    if (window->drawAllTrajectories)
    {
        f << ",variation";
    }
    f << "\n";

    if (isTrajectoryMode)
    {
        uint64_t totalVariations = comp.marshal.totalVariations;
        uint64_t variationsToDraw = window->drawAllTrajectories ? totalVariations : 1;

        for (uint64_t v = 0; v < variationsToDraw; ++v)
        {
            uint64_t variation = v;
            if (!window->drawAllTrajectories && !attributeValueIndices.empty())
            {
                uint64_t tempVariation = 0;
                steps2Variation(&tempVariation, attributeValueIndices.data(), &KERNEL);
                variation = tempVariation;
            }

            numb* trajectory = comp.marshal.trajectory + (comp.marshal.variationSize * variation);

            for (int step = 0; step <= steps; ++step)
            {
                numb* point = trajectory + ((unsigned long long)step * (unsigned long long)KERNEL.VAR_COUNT);

                f << point[xIdx] << "," << point[yIdx];

                if (window->drawAllTrajectories)
                {
                    f << "," << variation;
                }

                f << "\n";
            }
        }
    }
    else
    {
        extern int particleStep;

        uint64_t totalVariations = comp.marshal.totalVariations;
        int currentStep = particleStep;
        if (currentStep > KERNEL.steps) currentStep = KERNEL.steps;

        for (uint64_t v = 0; v < totalVariations; ++v)
        {
            numb* point = comp.marshal.trajectory +
                (comp.marshal.variationSize * v) +
                ((unsigned long long)currentStep * (unsigned long long)KERNEL.VAR_COUNT);

            f << point[xIdx] << "," << point[yIdx];

            f << "," << v;

            f << "\n";
        }
    }

    return path;
}

// ======================================================
//                3D PHASE DIAGRAM EXPORT
// ======================================================
std::string exportPhase3DCSV(const PlotWindow* window)
{
    if (!window)
    {
        printf("[export] ERROR: window is null\n");
        return {};
    }

    if (window->type != Phase)
    {
        printf("[export] ERROR: window is not Phase (3D diagram)\n");
        return {};
    }

    if (window->variables.size() < 3)
    {
        printf("[export] ERROR: Not enough variables for 3D phase diagram (need at least 3)\n");
        return {};
    }

    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
    {
        printf("[export] ERROR: Computation not ready\n");
        return {};
    }

    const int steps = computedSteps;
    if (steps <= 0)
    {
        printf("[export] ERROR: No steps computed\n");
        return {};
    }

    int xIdx = window->variables[0];
    int yIdx = window->variables[1];
    int zIdx = window->variables[2];

    if (xIdx < 0 || xIdx >= KERNEL.VAR_COUNT ||
        yIdx < 0 || yIdx >= KERNEL.VAR_COUNT ||
        zIdx < 0 || zIdx >= KERNEL.VAR_COUNT)
    {
        printf("[export] ERROR: Invalid variable indices (x=%d, y=%d, z=%d, VAR_COUNT=%d)\n",
            xIdx, yIdx, zIdx, KERNEL.VAR_COUNT);
        return {};
    }

    extern bool enabledParticles;
    extern int particleStep;

    bool isTrajectoryMode = (!enabledParticles);

    const std::string systemName = safe_system_name(KERNEL);
    std::string mode = isTrajectoryMode ? "trajectory" : "particles";
    std::string implotMode = window->isImplot3d ? "implot3d" : "phase3d";

    std::string path = build_export_path(systemName, implotMode + "_" + mode, "", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open())
    {
        printf("[export] ERROR: Cannot open file %s\n", path.c_str());
        return {};
    }

    f << KERNEL.variables[xIdx].name << ","
        << KERNEL.variables[yIdx].name << ","
        << KERNEL.variables[zIdx].name;

    if (window->drawAllTrajectories)
    {
        f << ",variation";
    }
    f << "\n";

    if (isTrajectoryMode)
    {
        uint64_t totalVariations = comp.marshal.totalVariations;
        uint64_t variationsToDraw = window->drawAllTrajectories ? totalVariations : 1;

        for (uint64_t v = 0; v < variationsToDraw; ++v)
        {
            uint64_t variation = v;
            if (!window->drawAllTrajectories && !attributeValueIndices.empty())
            {
                uint64_t tempVariation = 0;
                steps2Variation(&tempVariation, attributeValueIndices.data(), &KERNEL);
                variation = tempVariation;
            }

            numb* trajectory = comp.marshal.trajectory + (comp.marshal.variationSize * variation);

            for (int step = 0; step <= steps; ++step)
            {
                numb* point = trajectory + ((unsigned long long)step * (unsigned long long)KERNEL.VAR_COUNT);

                f << point[xIdx] << "," << point[yIdx] << "," << point[zIdx];

                if (window->drawAllTrajectories)
                {
                    f << "," << variation;
                }

                f << "\n";
            }
        }
    }
    else
    {
        uint64_t totalVariations = comp.marshal.totalVariations;
        int currentStep = particleStep;
        if (currentStep > KERNEL.steps) currentStep = KERNEL.steps;

        for (uint64_t v = 0; v < totalVariations; ++v)
        {
            numb* point = comp.marshal.trajectory +
                (comp.marshal.variationSize * v) +
                ((unsigned long long)currentStep * (unsigned long long)KERNEL.VAR_COUNT);

            f << point[xIdx] << "," << point[yIdx] << "," << point[zIdx];

            f << "," << v;

            f << "\n";
        }
    }

    return path;
}

// ======================================================
//                RGB HEATMAP EXPORT (MCHeatmap)
// ======================================================
std::string exportMCHeatmapCSV(const PlotWindow* window)
{
    if (!window)
    {
        printf("[export] ERROR: window is null\n");
        return {};
    }

    if (window->type != MCHeatmap)
    {
        printf("[export] ERROR: window is not MCHeatmap\n");
        return {};
    }

    if (window->variables.size() < 3)
    {
        printf("[export] ERROR: Not enough channels for RGB heatmap (need at least 3)\n");
        return {};
    }

    bool isHires = false;
    if (window->variables.size() > 0)
    {
        AnalysisIndex mapIdx = (AnalysisIndex)window->variables[0];
        isHires = (mapIdx == hiresIndex) || window->isFrozenAsHires;
    }

    const HeatmapProperties* heatmap = isHires ? &window->hireshmp : &window->hmp;
    Kernel* krnl = isHires ? &kernelHiresComputed : &(KERNEL);

    if (!heatmap)
    {
        printf("[export] ERROR: HeatmapProperties is null\n");
        return {};
    }

    HeatmapSizing sizing;
    sizing.loadPointers(krnl, const_cast<HeatmapProperties*>(heatmap));
    sizing.initValues();

    if (sizing.xSize <= 0 || sizing.ySize <= 0)
    {
        printf("[export] ERROR: Invalid heatmap dimensions (x=%d, y=%d)\n", sizing.xSize, sizing.ySize);
        return {};
    }

    for (int ch = 0; ch < 3; ch++)
    {
        AnalysisIndex channelIdx = (AnalysisIndex)window->variables[ch];
        if (channelIdx == -1) continue;

        if (!heatmap->channel[ch].valueBuffer)
        {
            printf("[export] ERROR: Channel %d buffer is null\n", ch);
            return {};
        }
    }

    const std::string systemName = safe_system_name(*krnl);
    std::string hiresSuffix = isHires ? "_hires" : "";

    std::string ch0Name = "R";
    std::string ch1Name = "G";
    std::string ch2Name = "B";

    AnalysisIndex ch0Idx = (AnalysisIndex)window->variables[0];
    AnalysisIndex ch1Idx = (AnalysisIndex)window->variables[1];
    AnalysisIndex ch2Idx = (AnalysisIndex)window->variables[2];

    if (ch0Idx != -1 && indices.find(ch0Idx) != indices.end()) ch0Name = indices[ch0Idx].name;
    if (ch1Idx != -1 && indices.find(ch1Idx) != indices.end()) ch1Name = indices[ch1Idx].name;
    if (ch2Idx != -1 && indices.find(ch2Idx) != indices.end()) ch2Name = indices[ch2Idx].name;

    std::string path = build_export_path(systemName, "rgbheatmap" + hiresSuffix, "", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open())
    {
        printf("[export] ERROR: Cannot open file %s\n", path.c_str());
        return {};
    }

    const std::string nameX = axis_name(*krnl, heatmap->typeX, heatmap->indexX);
    const std::string nameY = axis_name(*krnl, heatmap->typeY, heatmap->indexY);

    f << nameY << "\\" << nameX;
    for (int x = 0; x < sizing.xSize; ++x)
    {
        numb xValue = sizing.minX + sizing.stepX * (numb)x;
        f << ',' << xValue;
    }
    f << '\n';

    for (int y = 0; y < sizing.ySize; ++y)
    {
        numb yValue = sizing.minY + sizing.stepY * (numb)y;
        f << yValue;

        for (int x = 0; x < sizing.xSize; ++x)
        {
            size_t index = (size_t)y * (size_t)sizing.xSize + (size_t)x;

            numb rVal = (ch0Idx != -1) ? heatmap->channel[0].valueBuffer[index] : 0;
            numb gVal = (ch1Idx != -1) ? heatmap->channel[1].valueBuffer[index] : 0;
            numb bVal = (ch2Idx != -1) ? heatmap->channel[2].valueBuffer[index] : 0;

            // Формируем строку "r|g|b"
            f << ',' << rVal << '|' << gVal << '|' << bVal;
        }
        f << '\n';
    }

    return path;
}

// ======================================================
//                METRIC EXPORT
// ======================================================
std::string exportMetricCSV(const PlotWindow* window)
{
    if (!window) {
        printf("[export] ERROR: window is null\n");
        return {};
    }
    if (window->type != Metric) {
        printf("[export] ERROR: window is not Metric\n");
        return {};
    }
    Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.maps) {
        printf("[export] ERROR: computation not ready or maps not available\n");
        return {};
    }
    Kernel* krnl = &(KERNEL);
    Attribute* axis = window->typeX == MDT_Variable
        ? &(krnl->variables[window->indexX])
        : &(krnl->parameters[window->indexX]);

    if (axis->TrueStepCount() <= 1) {
        printf("[export] ERROR: Axis %s is fixed (no ranging)\n", axis->name.c_str());
        return {};
    }
    const std::vector<int>& selectedIndices = window->variables;
    if (selectedIndices.empty()) {
        printf("[export] ERROR: no indices selected for export\n");
        return {};
    }
    const std::string systemName = safe_system_name(*krnl);
    const std::string axisName = make_safe(axis->name);
    const std::string path = build_export_path(systemName, "index_" + axisName, /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) {
        printf("[export] ERROR: cannot open file %s\n", path.c_str());
        return {};
    }
    f << axis->name;
    for (int idx : selectedIndices) {
        auto it = indices.find((AnalysisIndex)idx);
        if (it != indices.end())
            f << ',' << it->second.name;
        else
            f << ",unknown";
    }
    f << '\n';
    uint64_t variation = 0;
    std::vector<int> tempAVI = attributeValueIndices;
    const int axisOffset = (window->typeX == MDT_Variable) ? 0 : krnl->VAR_COUNT;
    const int axisLocalIndex = window->indexX + axisOffset;
    for (int step = 0; step < axis->stepCount; ++step) {
        tempAVI[axisLocalIndex] = step;
        steps2Variation(&variation, tempAVI.data(), krnl);
        numb xValue = axis->min + axis->step * (numb)step;
        f << xValue;
        for (int idx : selectedIndices) {
            AnalysisIndex mapIdx = (AnalysisIndex)idx;
            auto* port = index2port(comp.marshal.kernel.analyses, mapIdx);
            if (!port) { f << ','; continue; }
            const numb* mapSlice = comp.marshal.maps + port->offset * comp.marshal.totalVariations;
            f << ',' << mapSlice[variation];
        }
        f << '\n';
    }

    return path;
}