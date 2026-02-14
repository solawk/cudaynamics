#pragma once
#include "../json/json.h"
#include <fstream>
#include <string>
#include "gui/commonItemDialogs.h"

bool JSONRead(std::string filename, json::jobject* out, bool dialog);

void JSONWrite(json::jobject obj, std::string filename, bool dialog);