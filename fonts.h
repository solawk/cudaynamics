#pragma once
#include <string>
#include <map>
#include <vector>
#include "imgui/backends/imgui_impl_win32.h"
#include "imgui/backends/imgui_impl_dx11.h"
#include "plotWindow.h"

enum FontStyle { FS_REGULAR, FS_BOLD, FS_ITALIC, FS_BOLD_ITALIC };

struct FontFamily
{
    std::string name;
    std::map<int, ImFont* [4]> sizeVariants;
};

extern std::vector<int> fontSizes;

extern std::vector<FontFamily> loadedFontFamilies;

extern FontSettings GlobalFontSettings;
extern bool fontNotDefault;

ImFont* GetFont(int familyIndex, int targetSize, bool bold, bool italic);
void FontMenu(PlotWindow* window);
void FontSizeSelector(int* currentSize, const std::vector<int>& sizes);
void FontLoading(ImGuiIO& io);