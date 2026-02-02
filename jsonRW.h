#pragma once
#include "../json/json.h"
#include <fstream>
#include <string>

bool JSONRead(std::string filename, json::jobject* out);

void JSONWrite(json::jobject obj, std::string filename);