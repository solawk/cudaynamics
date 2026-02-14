#pragma once
#include "jsonRW.h"

bool JSONRead(std::string filename, json::jobject* out, bool dialog)
{
	std::string content;
	bool userCancelled = false;

	if (!dialog)
	{
		std::ifstream fs(filename, std::ios::in);
		if (!fs.is_open()) return false;
		for (std::string line; getline(fs, line); ) content += line + '\n';
		fs.close();
	}
	else
	{
		if (!CommonItemDialogLoad(&content, &userCancelled))
		{
			if (!userCancelled) MessageBoxA(guiHwnd, "Configuration load failed!", "Error", MB_ICONERROR | MB_OK);
			return false;
		}
	}

	json::jobject jsonResult = json::jobject::parse(content);

	*out = jsonResult;
	return true;
}

void JSONWrite(json::jobject obj, std::string filename, bool dialog)
{
	std::string exportedJSON = obj.pretty();

	if (!dialog)
	{
		std::ofstream fs(filename, std::ios::out);
		fs << exportedJSON;
		fs.close();
	}
	else
	{
		if (!CommonItemDialogSave(exportedJSON))
		{
			MessageBoxA(guiHwnd, "Configuration save failed!", "Error", MB_ICONERROR | MB_OK);
		}
	}
}