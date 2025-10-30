#pragma once
#include "objects.h"
#include <string>
#include <fstream>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <string>
#include <fstream>
#include <locale>
#include <vector>

#include "heatmapSizing_struct.h"
#include "kernel_struct.h"
#include "main.h"
#include "main_utils.h"
#include "mapData_struct.h"
#include "heatmapProperties.hpp"
#include "computation_struct.h"

void exportToFile(std::string name, numb* values, int count);

std::string exportHeatmapCSV(const std::string& mapName,
    const HeatmapSizing& sizing,
    const HeatmapProperties* heatmap);

std::string exportTimeSeriesCSV(const PlotWindow* window);