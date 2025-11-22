#include "fonts.h"

std::vector<int> fontSizes = 
{
	8, 9, 10, 11, 12, 14, 16, 18,
    20, 22, 24, 26, 28, 36, 48, 72
};

std::vector<FontFamily> loadedFontFamilies;

FontSettings GlobalFontSettings(0, false, false, 24);

bool fontNotDefault = false;

ImFont* GetFont(int familyIndex, int targetSize, bool bold, bool italic)
{
    FontFamily& family = loadedFontFamilies[familyIndex];
    int bestSize = fontSizes[0];
    int minDiff = 1000;
    for (const auto& pair : family.sizeVariants)
    {
        int diff = abs(pair.first - targetSize);
        if (diff < minDiff)
        {
            minDiff = diff;
            bestSize = pair.first;
        }
    }

    ImFont* const* fonts = family.sizeVariants.at(bestSize);

    ImFont* f = fonts[bold ? (italic ? FS_BOLD_ITALIC : FS_BOLD) : (italic ? FS_ITALIC : FS_REGULAR)];
    if (!f) f = fonts[FS_REGULAR];

    return f;
}

// Рисует поле ввода размера шрифта в стиле Word
void FontSizeSelector(int* currentSize, const std::vector<int>& sizes)
{
    // 1. Поле ввода (Input)
    ImGui::SetNextItemWidth(50);
    ImGui::InputInt("##FontSizeInput", currentSize, 0, 0);

    // 2. Кнопка выпадающего списка (Combo), приклеенная справа
    ImGui::SameLine(0, 0); // Убираем отступ между элементами

    // ImGuiComboFlags_NoPreview означает, что мы рисуем только стрелочку, без текста внутри
    if (ImGui::BeginCombo("##FontSizeCombo", "", ImGuiComboFlags_NoPreview | ImGuiComboFlags_PopupAlignLeft))
    {
        for (int sz : sizes)
        {
            bool is_selected = (*currentSize == sz);

            std::string label = std::to_string((int)sz);
            if (ImGui::Selectable(label.c_str(), is_selected))
            {
                *currentSize = sz;
            }

            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}

void FontMenu(PlotWindow* window)
{
    FontSettings& settings = window == nullptr ? GlobalFontSettings : window->localFontSettings;

    if (!loadedFontFamilies.empty())
    {
        if (ImGui::BeginCombo("##Family", loadedFontFamilies[settings.family].name.c_str()))
        {
            for (int n = 0; n < loadedFontFamilies.size(); n++)
            {
                if (ImGui::Selectable(loadedFontFamilies[n].name.c_str(), settings.family == n))
                    settings.family = n;
            }
            ImGui::EndCombo();
        }

        FontSizeSelector(&settings.size, fontSizes);

        ImGui::Checkbox("Bold", &settings.isBold);
        ImGui::SameLine();
        ImGui::Checkbox("Italic", &settings.isItalic);
    }
    else
    {
        ImGui::TextDisabled("No fonts loaded!");
    }
}

void LoadFont(std::string name, std::string path1, std::string path2, ImGuiIO& io)
{
    FontFamily family;
    family.name = name;

    for (int size : fontSizes)
    {
        for (int s = FS_REGULAR; s < FS_BOLD_ITALIC; s++)
        {
            family.sizeVariants[size][s] = nullptr;
        }

        family.sizeVariants[size][FS_REGULAR]       = io.Fonts->AddFontFromFileTTF((path1 + path2).c_str(), (float)size);
        family.sizeVariants[size][FS_BOLD]          = io.Fonts->AddFontFromFileTTF((path1 + "-Bold" + path2).c_str(), (float)size);
        family.sizeVariants[size][FS_ITALIC]        = io.Fonts->AddFontFromFileTTF((path1 + "-Italic" + path2).c_str(), (float)size);
        family.sizeVariants[size][FS_BOLD_ITALIC]   = io.Fonts->AddFontFromFileTTF((path1 + "-BoldItalic" + path2).c_str(), (float)size);

        if (!family.sizeVariants[size][FS_BOLD])            family.sizeVariants[size][FS_BOLD] = family.sizeVariants[size][FS_REGULAR];
        if (!family.sizeVariants[size][FS_ITALIC])          family.sizeVariants[size][FS_ITALIC] = family.sizeVariants[size][FS_REGULAR];
        if (!family.sizeVariants[size][FS_BOLD_ITALIC])     family.sizeVariants[size][FS_BOLD_ITALIC] = family.sizeVariants[size][FS_BOLD];
    }

    if (family.sizeVariants[fontSizes[0]][FS_REGULAR])
        loadedFontFamilies.push_back(family);
    else
        printf(("Font " + name + " failed to load!\n").c_str());
}

void FontLoading(ImGuiIO& io)
{
    LoadFont("Ubuntu Mono", "fonts/UbuntuMono", ".ttf", io);
    LoadFont("Times New Roman", "fonts/TimesNewRoman", ".ttf", io);

    int defaultSize = 24;
    if (!loadedFontFamilies.empty())
    {
        if (loadedFontFamilies[0].sizeVariants.count(defaultSize))
            io.FontDefault = loadedFontFamilies[0].sizeVariants[defaultSize][FS_REGULAR];
        else
            io.FontDefault = loadedFontFamilies[0].sizeVariants.begin()->second[FS_REGULAR];

        fontNotDefault = true;
    }
    else
    {
        printf("No fonts have been loaded!\n");
    }
}