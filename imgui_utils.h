#pragma once
#include "imgui_main.h"
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

void populateAxisBuffer(float* buffer, float x, float y, float z);
void rotateOffsetBuffer2(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, float pitch, float yaw, ImVec4 offset, ImVec4 scale);
void rotateOffsetBuffer(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, ImVec4 rotation, ImVec4 offset, ImVec4 scale);

void populateRulerBuffer(float* buffer, float s, int dim);

void populateGridBuffer(float* buffer);
void gridX2Y(float* buffer);
void gridY2Z(float* buffer);

ImVec4 ToEulerAngles(ImVec4 q);

void cutoff2D(float* data, float* dst, int width, int height, int minX, int minY, int maxX, int maxY);
void compress2D(float* data, float* dst, int width, int height, int stride);

void getMinMax(float* data, int size, float* min, float* max);
void getMinMax2D(float* data, int size, ImVec2* min, ImVec2* max);