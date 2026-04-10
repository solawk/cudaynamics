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
        std::filesystem::create_directories("export_terminal");
        std::string system = launchConfig.has_key("system")
            ? (std::string)launchConfig["system"] : "hires";
        std::string index = launchConfig.has_key("index")
            ? (std::string)launchConfig["index"] : "matrix";
        basePath = "export_terminal/" + system + "_" + index;
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

    Marshal* marshal = &(computationHires.marshal);
    MarshalledKernel* kernel = &(marshal->kernel);
    numb* values = marshal->maps;
    int totalVariations = marshal->totalVariations;
    if (!values || totalVariations <= 0) { printf("FAIL: No data\n"); return; }

    struct AxisInfo 
    {
        bool isParameter;
        int index;
        int stepCount;
    };
    // Only for ranged axes
    std::vector<AxisInfo> rangedAxes;

    for (int i = 0; i < kernel->VAR_COUNT; ++i)
    {
        if (kernel->variables[i].rangingType != RT_None) rangedAxes.push_back({ false, i, kernel->variables[i].TrueStepCount() });
    }
    for (int i = 0; i < kernel->PARAM_COUNT; ++i)
    {
        if (kernel->parameters[i].rangingType != RT_None) rangedAxes.push_back({ true, i, kernel->parameters[i].TrueStepCount() });
    }

    std::sort(rangedAxes.begin(), rangedAxes.end(), [](const AxisInfo& a, const AxisInfo& b) 
        {
            return a.stepCount > b.stepCount;
        });

    std::string nameX = "-", nameY = "-";
    int cols = 1, rows = 1;

    if (rangedAxes.size() >= 1)
    {
        nameX = !rangedAxes[0].isParameter ? kernel->variables[rangedAxes[0].index].name : kernel->parameters[rangedAxes[0].index].name;
        cols = rangedAxes[0].stepCount;
    }
    if (rangedAxes.size() >= 2)
    {
        nameY = !rangedAxes[1].isParameter ? kernel->variables[rangedAxes[1].index].name : kernel->parameters[rangedAxes[1].index].name;
        rows = rangedAxes[1].stepCount;
    }

    int sliceCount = 1;
    for (int i = 2; i < rangedAxes.size(); i++) sliceCount *= rangedAxes[i].stepCount;

    // For ALL attributes
    std::vector<int> avi;
    for (int i = 0; i < kernel->VAR_COUNT + kernel->PARAM_COUNT; i++) avi.push_back(0);

    printf("Exporting: %s(%d) \\ %s(%d)", nameY.c_str(), rows, nameX.c_str(), cols);
    if (sliceCount > 1) printf(" x %d slices", sliceCount);
    printf(", %d total values\n", totalVariations);

    for (int s = 0; s < sliceCount; s++)
    {
        //printf("avi: ");
        //for (int i = 0; i < kernel->VAR_COUNT + kernel->PARAM_COUNT; i++) printf("%i_", avi[i]);
        //printf("\n");

        std::string sliceAVIstring = "";
        for (int a = 2; a < rangedAxes.size(); a++)
        {
            int aIndex = !rangedAxes[a].isParameter ? rangedAxes[a].index : kernel->VAR_COUNT + rangedAxes[a].index;
            sliceAVIstring += "_" + std::to_string(avi[aIndex]);
        }
        std::string fileName = std::to_string(s + 1) + "_" +
            basePath.substr(basePath.find_last_of("/\\") + 1) +
            "_" + std::to_string(rows) + "x" + std::to_string(cols) +
            sliceAVIstring + ".csv";
        std::string fullPath = "export_terminal/" + fileName;
        std::ofstream file(fullPath);
        if (!file.is_open()) { printf("FAIL: Can't open %s for writing\n", fullPath.c_str()); }
        file.imbue(std::locale::classic());
        file << std::fixed << std::setprecision(9);
        //printf("slice\n");

        numb* valuesOfAxis = !rangedAxes[0].isParameter ? kernel->variables[rangedAxes[0].index].values : kernel->parameters[rangedAxes[0].index].values;
        //printf("%s\\%s", nameY.c_str(), nameX.c_str());
        //for (int x = 0; x < cols; x++) printf(",%f", (float)valuesOfAxis[x]);
        //printf("\n");
        file << nameY << "\\" << nameX;
        for (int x = 0; x < cols; x++) file << ',' << valuesOfAxis[x];
        file << '\n';

        uint64_t variation;
        int* aviData = avi.data();
        int aviX = !rangedAxes[0].isParameter ? rangedAxes[0].index : rangedAxes[0].index + kernel->VAR_COUNT;
        int aviY = !rangedAxes[1].isParameter ? rangedAxes[1].index : rangedAxes[1].index + kernel->VAR_COUNT;

        valuesOfAxis = !rangedAxes[1].isParameter ? kernel->variables[rangedAxes[1].index].values : kernel->parameters[rangedAxes[1].index].values;
        for (int y = 0; y < rows; y++)
        {
            file << valuesOfAxis[y];
            //printf("%f", (float)valuesOfAxis[y]);
            for (int x = 0; x < cols; x++)
            {
                avi[aviX] = x;
                avi[aviY] = y;
                
                steps2Variation(&variation, aviData, kernel);

                file << ',' << values[variation];
                //printf(",%f", (float)values[variation]);
            }
            file << '\n';
            //printf("\n");
        }
        file.close();

        // Selecting next slice
        for (int a = (int)rangedAxes.size() - 1; a >= 2; a--)
        {
            int aIndex = !rangedAxes[a].isParameter ? rangedAxes[a].index : kernel->VAR_COUNT + rangedAxes[a].index;
            avi[aIndex]++;
            if (avi[aIndex] < rangedAxes[a].stepCount) break;
            avi[aIndex] = 0;
        }
    }
}