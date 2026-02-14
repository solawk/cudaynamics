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