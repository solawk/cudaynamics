#include "standalone_utils.h"

json::jobject launchConfig;
std::string exportPath = "";
bool launchedAsOneShot;

std::map<std::string, std::string> launchOptions(int argc, char** argv)
{
    std::map<std::string, std::string> options;

    for (int i = 1; i < argc; i += 2)
    {
        if (i == argc - 1) break; // Flag present but no option
        options[argv[i]] = argv[i + 1];
    }

    return options;
}

bool readLaunchOptions(int argc, char** argv)
{
    launchedAsOneShot = false;

    std::map<std::string, std::string> options = launchOptions(argc, argv);
    if (options.count("-c"))
    {
        if (JSONRead(options["-c"], &launchConfig, false))
        {
            printf(("Reading configuration file " + options["-c"] + "\n").c_str());
            launchedAsOneShot = true;
        }
        else
        {
            printf(("FAIL: Launch configuration file " + options["-c"] + " not found, aborting...\n").c_str());
            return false;
        }
    }

    if (options.count("-f"))
    {
        exportPath = options["-f"];
    }

    return true;
}

void exportHires()
{
    std::string basePath;
    if (launchedAsOneShot && exportPath.empty())
    {
        std::filesystem::create_directories("export_console");
        std::string system = launchConfig.has_key("system")
            ? (std::string)launchConfig["system"] : "hires";
        std::string index = launchConfig.has_key("index")
            ? (std::string)launchConfig["index"] : "matrix";
        basePath = "export_console/" + system + "_" + index;
    }
    else if (!exportPath.empty())
    {
        basePath = exportPath;
        if (basePath.size() >= 4 && basePath.substr(basePath.size() - 4) == ".txt")
            basePath = basePath.substr(0, basePath.size() - 4);
    }
    else
    {
        basePath = "hiresOutput";
    }

    auto& m = computationHires.marshal;
    numb* values = m.maps;
    int total = m.totalVariations;
    if (!values || total <= 0) { printf("FAIL: No data\n"); return; }

    struct AxisInfo {
        std::string name;
        int stepCount;
        numb min;
        numb step;
    };
    std::vector<AxisInfo> axes;
    auto& k = m.kernel;

    auto collectAxis = [&](Attribute& attr) {
        if (attr.rangingType != RT_None) {
            axes.push_back({ attr.name, attr.TrueStepCount(), attr.min, attr.step });
        }
        };
    for (int i = 0; i < k.VAR_COUNT; ++i) collectAxis(k.variables[i]);
    for (int i = 0; i < k.PARAM_COUNT; ++i) collectAxis(k.parameters[i]);

    std::sort(axes.begin(), axes.end(), [](const AxisInfo& a, const AxisInfo& b) {
        return a.stepCount > b.stepCount;
        });

    std::string nameX = "X", nameY = "Y";
    numb minX = 0, stepX = 1, minY = 0, stepY = 1;
    int cols = 1, rows = 1;

    std::vector<AxisInfo> sliceAxes;
    for (size_t i = 2; i < axes.size(); ++i) {
        if (axes[i].stepCount > 1) {
            sliceAxes.push_back(axes[i]);
        }
    }

    if (axes.size() >= 1) { nameX = axes[0].name; minX = axes[0].min; stepX = axes[0].step; cols = axes[0].stepCount; }
    if (axes.size() >= 2) {
        nameY = axes[1].name; minY = axes[1].min; stepY = axes[1].step; rows = axes[1].stepCount;
        if (axes[0].stepCount > axes[1].stepCount) {
            std::swap(nameX, nameY); std::swap(minX, minY); std::swap(stepX, stepY); std::swap(cols, rows);
        }
    }

    if (axes.size() < 2) {
        for (int a = static_cast<int>(sqrt(total)); a >= 1; --a)
            if (total % a == 0) { rows = a; cols = total / a; break; }
        printf("INFO: Fallback dimensions %dx%d\n", rows, cols);
    }

    int sliceCount = 1;
    for (const auto& sa : sliceAxes) { sliceCount *= sa.stepCount; }

    int expectedTotal = rows * cols * sliceCount;
    if (expectedTotal != total) {
        printf("WARNING: Expected %d values, got %d. Adjusting...\n", expectedTotal, total);
        if (!sliceAxes.empty()) {
            if (total % (rows * cols) == 0) {
                sliceCount = total / (rows * cols);
                printf("Adjusted sliceCount to %d\n", sliceCount);
            }
        }
    }

    printf("Exporting: %s(%d) \\ %s(%d)", nameY.c_str(), rows, nameX.c_str(), cols);
    if (!sliceAxes.empty()) printf(" x %d slices", sliceCount);
    printf(", %d total values\n", total);

    auto exportMatrixSlice = [&](int sliceIdx, const std::string& filePath) {
        std::ofstream file(filePath);
        if (!file.is_open()) { printf("FAIL: Can't open %s\n", filePath.c_str()); return false; }

        file << nameY << "\\" << nameX;
        for (int x = 0; x < cols; ++x)
            file << ',' << (minX + stepX * static_cast<numb>(x));
        file << '\n';

        size_t sliceOffset = static_cast<size_t>(sliceIdx) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
        for (int y = 0; y < rows; ++y) {
            file << (minY + stepY * static_cast<numb>(y));
            size_t base = sliceOffset + static_cast<size_t>(y) * static_cast<size_t>(cols);
            for (int x = 0; x < cols; ++x)
                file << ',' << values[base + x];
            file << '\n';
        }
        file.close();
        return true;
        };

    if (!sliceAxes.empty())
    {
        for (int s = 0; s < sliceCount; ++s)
        {
            std::string sliceIndices = "";
            int remaining = s;
            for (int i = (int)sliceAxes.size() - 1; i >= 0; --i) {
                int idx = remaining % sliceAxes[i].stepCount;
                sliceIndices = "_" + std::to_string(idx) + sliceIndices;
                remaining /= sliceAxes[i].stepCount;
            }

            std::string fileName = std::to_string(s + 1) + "_" +
                basePath.substr(basePath.find_last_of("/\\") + 1) +
                "_" + std::to_string(rows) + "x" + std::to_string(cols) +
                sliceIndices + ".csv";
            std::string fullPath = "export_console/" + fileName;

            if (exportMatrixSlice(s, fullPath))
                printf("[%d/%d] Exported %s\n", s + 1, sliceCount, fileName.c_str());
        }
    }
    else
    {
        std::string path = basePath +
            "_" + std::to_string(rows) + "x" + std::to_string(cols) +
            ".csv";
        if (exportMatrixSlice(0, path))
            printf("Exported %s\n", path.c_str());
    }
}