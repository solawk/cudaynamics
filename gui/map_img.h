#pragma once
#include "../imgui_main.hpp"
#include <chrono>

void MapToImg(numb* mapBuffer, unsigned char** dataBuffer, int width, int height, numb min, numb max, ImPlotColormap colormap);

void MultichannelMapToImg(HeatmapProperties* heatmap, unsigned char** dataBuffer, int width, int height, bool ch0, bool ch1, bool ch2);