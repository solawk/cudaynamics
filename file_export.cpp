#include "file_export.h"

extern Computation computations[2];
extern int playedBufferIndex;
extern int computedSteps;
extern int bufferNo;
extern std::vector<int> attributeValueIndices;
extern numb getStepSize(Kernel& kernel);
extern void steps2Variation(int* variationIndex, int* avi, Kernel* kernel);

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

    // Создаёт директорию export/ и собирает путь. extra может быть пустым.
    std::string build_export_path(std::string_view systemName,
        std::string_view artifactName,   // например: "timeSeries" или имя карты
        std::string_view extra,          // например: суффикс или пусто
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

    // Имя оси для heatmap
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

    // Унифицированное имя системы
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
    f << nameX << "|" << nameY;
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

    // Активное вычисление
    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
        return {};

    // Проверки шагов и переменных
    const int steps = computedSteps;
    if (steps <= 0 || KERNEL.VAR_COUNT <= 0)
        return {};

    // Индекс вариации
    int variation = 0;
    if (!attributeValueIndices.empty())
        steps2Variation(&variation, attributeValueIndices.data(), &KERNEL);

    const unsigned long long varStride = comp.marshal.variationSize;
    const numb* base = comp.marshal.trajectory + varStride * (unsigned long long)variation;

    // Имя файла
    const std::string systemName = safe_system_name(KERNEL);
    const std::string path = build_export_path(systemName, "timeSeries", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    // Ось X
    const bool useTime = KERNEL.usingTime;
    const numb stepSize = getStepSize(KERNEL);
    const double start = (double)(bufferNo * KERNEL.steps + KERNEL.transientSteps) *
        (useTime ? (double)stepSize : 1.0);

    // === Заголовок ===
    f << (useTime ? "time" : "step");
    for (int v : window->variables) {
        if (v >= 0 && v < KERNEL.VAR_COUNT)
            f << ',' << KERNEL.variables[v].name;
        else
            f << ',';
    }
    f << '\n';

    // === Данные ===
    const int rows = steps + 1;
    for (int i = 0; i < rows; ++i) {
        const double x = start + (useTime ? i * (double)stepSize : i);
        f << x;

        const numb* row = base + (unsigned long long)i * (unsigned long long)KERNEL.VAR_COUNT;
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

    // Активное вычисление
    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
        return {};

    // Проверки шагов и переменных
    const int steps = computedSteps;
    if (steps <= 0 || KERNEL.VAR_COUNT <= 0)
        return {};

    // Имя файла
    const std::string systemName = safe_system_name(KERNEL);
    const std::string path = build_export_path(systemName, "decay", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    // Ось X
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
    const int rows = (int)window->decayBuffer[0].size();
    for (int i = 0; i < rows; ++i) 
    {
        const double x = window->decayBuffer[0][i];
        f << x;

        for (int t = 0; t < thresholdCount; t++)
        {
            f << ',' << window->decayAlive[t][i];
        }

        f << ',' << window->decayTotal[0][i] << '\n';
    }

    return path;
}

// ======================================================
//                ORBIT EXPORT
// ======================================================
std::string exportOrbitCSV(const PlotWindow* window)
{
    if (!window) return {};

    // Активное вычисление
    const Computation& comp = computations[playedBufferIndex];
    if (!comp.ready || !comp.marshal.trajectory)
        return {};

    // Проверки шагов и переменных
    const int steps = computedSteps;
    if (steps <= 0 || KERNEL.VAR_COUNT <= 0)
        return {};

    if (window->OrbitType == Selected_Var_Section) return {};

    // Имя файла
    const std::string systemName = safe_system_name(KERNEL);
    const std::string path = build_export_path(systemName, "orbit", /*extra*/"", ".csv");
    std::ofstream f = open_csv(path);
    if (!f.is_open()) return {};

    Attribute* axis = &(KERNEL.parameters[window->OrbitXIndex]);

    switch (window->OrbitType)
    {
    case Peak_Bifurcation:
        f << axis->name << ",Peaks" << '\n';
        for (int i = 0; i < window->BifDotAmount; ++i)
        {
            f << window->bifParamIndices[i] << "," << window->bifAmps[i] << '\n';
        }
        break;
    case Interval_Bifurcation:
        f << axis->name << ",Intervals" << '\n';
        for (int i = 0; i < window->BifDotAmount; ++i)
        {
            f << window->bifParamIndices[i] << "," << window->bifIntervals[i] << '\n';
        }
        break;
    case Bifurcation_3D:
        f << axis->name << ",Peaks,Intervals" << '\n';
        for (int i = 0; i < window->BifDotAmount; ++i)
        {
            f << window->bifParamIndices[i] << "," << window->bifAmps[i] << "," << window->bifIntervals[i] << '\n';
        }
        break;
    }

    return path;
}