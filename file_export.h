#pragma once
#include <fstream>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <vector>

#include "heatmapSizing_struct.h"
#include "main.h"
#include "heatmapProperties.hpp"

void exportToFile(std::string name, numb* values, int count);

std::string exportHeatmapCSV(const std::string& mapName,
    const HeatmapSizing& sizing,
    const HeatmapProperties* heatmap);

std::string exportTimeSeriesCSV(const PlotWindow* window);

std::string exportDecayCSV(const PlotWindow* window);

std::string exportOrbitCSV(const PlotWindow* window);