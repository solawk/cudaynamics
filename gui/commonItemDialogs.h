#pragma once
#include <windows.h>
#include <shobjidl.h> 
#include <shlwapi.h>
#include <iostream>
#include <fstream>
#include <string>
#include "gui/hwnd.h"
#include "../kernel_map.h"

bool CommonItemDialogSave(std::string& content);

bool CommonItemDialogLoad(std::string* content, bool* userCancelled);