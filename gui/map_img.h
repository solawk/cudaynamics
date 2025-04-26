#pragma once
#include "../imgui_main.hpp"
#include <chrono>

void MapToImg(numb* mapBuffer, unsigned char** dataBuffer, int width, int height, numb min, numb max);
