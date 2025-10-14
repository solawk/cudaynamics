#pragma once

#include "imgui_main.hpp"

void listAttrRanging(Attribute* attr, bool isChanged);

void listAttrNumb(Attribute* attr, numb* field, std::string name, std::string inner, bool isChanged);

void listAttrInt(Attribute* attr, int* field, std::string name, std::string inner, bool isChanged);

void listVariable(int i);

void listParameter(int i);

void listEnum(int i);

void mapSelectionCombo(std::string name, int& selectedIndex, bool addEmpty);