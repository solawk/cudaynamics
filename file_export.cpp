#include "file_export.h"

void exportToFile(std::string name, numb* values, int count)
{
    std::ofstream outputFileStream(name.c_str(), std::ios::out);
    std::string valueStr;

    for (int i = 0; i < count; i++)
    {
        valueStr = std::to_string(*(values + i));
        outputFileStream << valueStr << std::endl;
    }

    outputFileStream.close();
}