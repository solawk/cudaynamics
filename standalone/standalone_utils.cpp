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
        if (JSONRead(options["-c"], &launchConfig))
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

bool launchOneShotComputation()
{
    // Switch to system
    if (!launchConfig.has_key("system"))
    {
        printf("FAIL: System name not specified, aborting...\n");
        return false;
    }
    std::string systemName = (std::string)launchConfig["system"];
    if (!kernels.count(systemName))
    {
        printf(("FAIL: System " + systemName + " not present, aborting...\n").c_str());
        return false;
    }
    printf(("System " + systemName + " found\n").c_str());
    selectedKernel = systemName;

    // Enable the system
    prepareKernel();

    // Select analysis index
    if (launchConfig.has_key("index"))
    {
        std::string indexName = (std::string)launchConfig["index"];
        int index = -1;
        for (int i = 0; i < IND_COUNT; i++)
        {
            if (indices[(AnalysisIndex)i].name == indexName)
            {
                index = i;
                printf(("Index " + indexName + " found\n").c_str());
                break;
            }
        }
        if (index == -1)
        {
            printf(("FAIL: Index " + indexName + " not present, aborting...\n").c_str());
            return false;
        }

        hiresIndex = (AnalysisIndex)index;
    }
    else
    {
        printf("FAIL: Analysis index not specified, aborting...\n");
        return false;
    }

    // Setup the system
    if (launchConfig.has_key("steps")) KERNEL.steps = (int)launchConfig["steps"];
    if (launchConfig.has_key("transient")) KERNEL.transientSteps = (int)launchConfig["transient"];
    if (launchConfig.has_key("variables"))
    {
        int variableSettingsCount = launchConfig["variables"].array_size();
        for (int i = 0; i < variableSettingsCount; i++)
        {
            json::jobject varSettings = (json::jobject)launchConfig["variables"].array(i);
            std::string varName = (std::string)varSettings["name"];
            int varIndex = -1;
            for (int j = 0; j < KERNEL.VAR_COUNT && varIndex == -1; j++) if (KERNEL.variables[j].name == varName) varIndex = j;
            if (varIndex == -1)
            {
                printf(("FAIL: Variable " + varName + " not present, aborting...\n").c_str());
                return false;
            }
            if (varSettings.has_key("ranging")) KERNEL.variables[varIndex].rangingType = rangingTypeFromString(varSettings["ranging"]);
            if (varSettings.has_key("min")) KERNEL.variables[varIndex].min = (numb)varSettings["min"];
            if (varSettings.has_key("max")) KERNEL.variables[varIndex].max = (numb)varSettings["max"];
            if (varSettings.has_key("step")) KERNEL.variables[varIndex].step = (numb)varSettings["step"];
            if (varSettings.has_key("stepCount")) KERNEL.variables[varIndex].stepCount = (int)varSettings["stepCount"];
        }
    }
    if (launchConfig.has_key("parameters"))
    {
        int parameterSettingsCount = launchConfig["parameters"].array_size();
        for (int i = 0; i < parameterSettingsCount; i++)
        {
            json::jobject paramSettings = (json::jobject)launchConfig["parameters"].array(i);
            std::string paramName = (std::string)paramSettings["name"];
            int paramIndex = -1;
            for (int j = 0; j < KERNEL.PARAM_COUNT && paramIndex == -1; j++) if (KERNEL.parameters[j].name == paramName) paramIndex = j;
            if (paramIndex == -1)
            {
                printf(("FAIL: Parameter " + paramName + " not present, aborting...\n").c_str());
                return false;
            }
            if (paramSettings.has_key("ranging")) KERNEL.parameters[paramIndex].rangingType = rangingTypeFromString(paramSettings["ranging"]);
            if (paramSettings.has_key("min")) KERNEL.parameters[paramIndex].min = (numb)paramSettings["min"];
            if (paramSettings.has_key("max")) KERNEL.parameters[paramIndex].max = (numb)paramSettings["max"];
            if (paramSettings.has_key("step")) KERNEL.parameters[paramIndex].step = (numb)paramSettings["step"];
            if (paramSettings.has_key("stepCount")) KERNEL.parameters[paramIndex].stepCount = (int)paramSettings["stepCount"];
        }
    }
    if (launchConfig.has_key("analysis"))
    {
        int analysisSettingsCount = launchConfig["analysis"].array_size();
        for (int i = 0; i < analysisSettingsCount; i++)
        {
            json::jobject analysisSettings = (json::jobject)launchConfig["analysis"].array(i);
            std::string analysisName = (std::string)analysisSettings["name"];
            bool analysisFound = false;
            for (int j = 0; j < ANF_COUNT; j++)
            {
                if ((std::string)AnFuncNames[j] == analysisName)
                {
                    analysisFound = true;
                    std::vector<std::string> settings = analysisSettings["settings"].array_as_vector();

                    switch ((AnalysisFunction)j)
                    {
                    case AnalysisFunction::ANF_MINMAX:
                        KERNEL.analyses.MINMAX.setup(settings);
                        break;
                    case AnalysisFunction::ANF_LLE:
                        KERNEL.analyses.LLE.setup(settings);
                        break;
                    case AnalysisFunction::ANF_PERIOD:
                        KERNEL.analyses.PERIOD.setup(settings);
                        break;
                    case AnalysisFunction::ANF_PV:
                        KERNEL.analyses.PV.setup(settings);
                        break;
                    }
                    break;
                }
            }
            if (!analysisFound)
            {
                printf(("FAIL: Analysis " + analysisName + " not present, aborting...\n").c_str());
                return false;
            }
        }
    }
    printf("No setup problems occured\n");

    return true;
}

void exportHires()
{
    std::string path;
    if (exportPath == "")
    {
        path = "hiresOutput.txt";
    }
    else
    {
        path = exportPath;
    }
    printf(("Export path is " + path + "\n").c_str());
    
    std::ofstream filestream(path);
    if (!filestream.is_open())
    {
        printf("FAIL: File couldn't be opened\n");
        return;
    }
    numb* values = computationHires.marshal.maps;
    for (int i = 0; i < computationHires.marshal.totalVariations; ++i)
        filestream << values[i] << '\n';
    filestream.close();
}