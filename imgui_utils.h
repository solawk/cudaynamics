#pragma once
#include "imgui_main.hpp"
#include "implot/implot.h"
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <objects.h>
#include <vector>
#include <implot_internal.h>

const float DEG2RAD = 3.141592f / 180.0f;

std::string memoryString(unsigned long long bytes);
std::string scaleString(float scale);

void populateAxisBuffer(numb* buffer, float x, float y, float z);
void rotateOffsetBuffer(numb* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale);
void rotateOffsetBufferQuat(numb* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale);

void populateRulerBuffer(numb* buffer, float s, int dim);

void populateGridBuffer(numb* buffer);
void gridX2Y(numb* buffer);
void gridY2Z(numb* buffer);

ImVec4 ToEulerAngles(ImVec4 q);

void cutoff2D(numb* data, numb* dst, int width, int height, int minX, int minY, int maxX, int maxY);
//void compress2D(numb* data, numb* dst, int width, int height, int stride);

void getMinMax(numb* data, int size, numb* min, numb* max);
void getMinMax2D(numb* data, int size, ImVec2* min, ImVec2* max);

std::string padString(std::string str, int length);