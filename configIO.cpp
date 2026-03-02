#include "configIO.h"

json::jobject saveCfg(bool saveHires, bool saveNew)
{
    Kernel* k = !saveHires ? (!saveNew ? &KERNEL : &kernelNew) : (!saveNew ? &kernelHiresComputed : &kernelHiresNew);

    // Config (.cfg to not confuse with ImGui .config) stores the dynamical system settings and (in the future) window placement (like ImGui .config does now)
    // 1. selected system
    // 2. selected hi-res index
    // 3. steps/time mode
    // 4. steps/transient steps/time
    // 5. all attribute settings
    // 6. all analysis settings
    json::jobject cfg;
    cfg["system"] = selectedKernel; // 1
    cfg["index"] = hiresIndex != IND_NONE ? indices[hiresIndex].name : ""; // 2
    cfg["usingTime"].set_boolean(k->usingTime); // 3
    // 4
    if (k->usingTime)
    {
        cfg["time"] = k->time;
        cfg["transientTime"] = k->transientTime;
    }
    else
    {
        cfg["steps"] = k->steps;
        cfg["transientSteps"] = k->transientSteps;
    }
    // 5
    std::vector<json::jobject> variables, parameters;
    for (Attribute v : k->variables) variables.push_back(v.ExportToJSON());
    for (Attribute p : k->parameters) parameters.push_back(p.ExportToJSON());
    cfg["variables"] = variables;
    cfg["parameters"] = parameters;
    // 6
    std::vector<json::jobject> analysis;
    analysis.push_back(k->analyses.MINMAX.ExportSettings());
    analysis.push_back(k->analyses.LLE.ExportSettings());
    analysis.push_back(k->analyses.PERIOD.ExportSettings());
    analysis.push_back(k->analyses.PV.ExportSettings());
    cfg["analysis"] = analysis;

    return cfg;
}

bool loadCfg(json::jobject cfg, bool switchSystem, bool cleanStart, bool needPrints)
{
    if (switchSystem)
    {
        // Switch to system
        if (!cfg.has_key("system"))
        {
            printf("FAIL: System name not specified, aborting...\n");
            return false;
        }
        std::string systemName = (std::string)cfg["system"];
        if (!kernels.count(systemName))
        {
            printf(("FAIL: System " + systemName + " not present, aborting...\n").c_str());
            return false;
        }
        if (needPrints) printf(("System " + systemName + " found\n").c_str());
        selectedKernel = systemName;

        // Enable the system
        if (cleanStart) prepareKernel(); // Clean start means initializing the suite, which is not needed when loading the cfg via GUI
    }

    // Explicitly setting variationsPerParallelization
    if (cfg.has_key("varPerParallelization")) applicationSettings.varPerParallelization = (int)cfg["varPerParallelization"];

    // Select analysis index
    if (cfg.has_key("index"))
    {
        std::string indexName = (std::string)cfg["index"];

        if (indexName == "")
        {
            hiresIndex = IND_NONE;
        }
        else
        {
            int index = -1;
            for (int i = 0; i < IND_COUNT; i++)
            {
                if (indices[(AnalysisIndex)i].name == indexName)
                {
                    index = i;
                    if (needPrints) printf(("Index " + indexName + " found\n").c_str());
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
    }
    else
    {
        printf("FAIL: Analysis index not specified, aborting...\n");
        return false;
    }

    if (hiresIndex == IND_NONE)
    {
        kernelNew.CopyFrom(&KERNEL);
    }
    else
    {
        kernelHiresNew.CopyFrom(&KERNEL);
    }
    Kernel* k = (hiresIndex == IND_NONE ? &kernelNew : &kernelHiresNew);

    // Setup the system

    if (cfg.has_key("usingTime"))
    {
        if (cfg["usingTime"].is_true())
        {
            k->usingTime = true;
            k->time = (numb)cfg["time"];
            k->transientTime = (numb)cfg["transientTime"];
        }
        else
        {
            k->usingTime = false;
            k->steps = (int)cfg["steps"];
            k->transientSteps = (int)cfg["transientSteps"];
        }
    }

    if (cfg.has_key("variables"))
    {
        int variableSettingsCount = cfg["variables"].array_size();
        for (int i = 0; i < variableSettingsCount; i++)
        {
            json::jobject varSettings = (json::jobject)cfg["variables"].array(i);
            std::string varName = (std::string)varSettings["name"];
            int varIndex = -1;
            for (int j = 0; j < KERNEL.VAR_COUNT && varIndex == -1; j++) if (KERNEL.variables[j].name == varName) varIndex = j;
            if (varIndex == -1)
            {
                printf(("FAIL: Variable " + varName + " not present, aborting...\n").c_str());
                return false;
            }

            k->variables[varIndex].ImportFromJSON(varSettings);
        }
    }
    if (cfg.has_key("parameters"))
    {
        int parameterSettingsCount = cfg["parameters"].array_size();
        for (int i = 0; i < parameterSettingsCount; i++)
        {
            json::jobject paramSettings = (json::jobject)cfg["parameters"].array(i);
            std::string paramName = (std::string)paramSettings["name"];
            int paramIndex = -1;
            for (int j = 0; j < KERNEL.PARAM_COUNT && paramIndex == -1; j++) if (KERNEL.parameters[j].name == paramName) paramIndex = j;
            if (paramIndex == -1)
            {
                printf(("FAIL: Parameter " + paramName + " not present, aborting...\n").c_str());
                return false;
            }

            k->parameters[paramIndex].ImportFromJSON(paramSettings);
        }
    }
    if (cfg.has_key("analysis"))
    {
        int analysisSettingsCount = cfg["analysis"].array_size();
        for (int i = 0; i < analysisSettingsCount; i++)
        {
            json::jobject analysisSettings = (json::jobject)cfg["analysis"].array(i);
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
                        k->analyses.MINMAX.setup(settings);
                        break;
                    case AnalysisFunction::ANF_LLE:
                        k->analyses.LLE.setup(settings);
                        break;
                    case AnalysisFunction::ANF_PERIOD:
                        k->analyses.PERIOD.setup(settings);
                        break;
                    case AnalysisFunction::ANF_PV:
                        k->analyses.PV.setup(settings);
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
    if (needPrints) printf("No setup problems occured\n");

    return true;
}