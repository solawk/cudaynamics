#pragma once
#include "imgui_main.hpp"
#include "implot/implot.h"
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <objects.h>
#include <vector>
#include <implot_internal.h>
#include "attribute_struct.h"

const float DEG2RAD = 3.141592f / 180.0f;

std::string memoryString(unsigned long long bytes);
std::string scaleString(float scale);

void populateAxisBuffer(float* buffer, float x, float y, float z);
void rotateOffsetBuffer(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale);
void rotateOffsetBuffer(double* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale);
void rotateOffsetBufferQuat(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale);

void populateRulerBuffer(float* buffer, float s, int dim);

void populateGridBuffer(float* buffer);
void gridX2Y(float* buffer);
void gridY2Z(float* buffer);

ImVec4 ToEulerAngles(ImVec4 q);

void cutoff2D(numb* data, numb* dst, int width, int height, int minX, int minY, int maxX, int maxY);
//void compress2D(numb* data, numb* dst, int width, int height, int stride);

void getMinMax(numb* data, int size, numb* min, numb* max);
void getMinMax2D(numb* data, int size, ImVec2* min, ImVec2* max, int varCount);

std::string padString(std::string str, int length);

void addDeltaQuatRotation(PlotWindow* window, float deltax, float deltay);

// Get color of the plot/particle when heatmap-painting
int getVariationGroup(colorLUT* lut, int variation);

int findParameterByName(std::string name);
bool isEnumEnabledByString(Attribute& enumAttribute, std::string str);