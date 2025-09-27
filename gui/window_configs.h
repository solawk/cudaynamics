#pragma once

#include "imgui_main.hpp"

extern std::vector<PlotWindow> plotWindows;
extern int uniqueIds;

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