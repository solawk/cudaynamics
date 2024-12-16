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

void populateAxisBuffer(float* buffer, float x, float y, float z);

void rotateOffsetBuffer(float* buffer, int pointCount, int varCount, int xdo, int ydo, int zdo, float pitch, float yaw, ImVec4 offset, ImVec4 scale);

void populateGridBuffer(float* buffer);
void gridX2Y(float* buffer);
void gridY2Z(float* buffer);