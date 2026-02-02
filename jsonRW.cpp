#pragma once
#include "jsonRW.h"

bool JSONRead(std::string filename, json::jobject* out)
{
	std::ifstream fs(filename, std::ios::in);

	if (!fs.is_open())
	{
		return false;
	}

	std::string content;
	for (std::string line; getline(fs, line); )
	{
		content += line + '\n';
	}

	fs.close();

	json::jobject jsonResult = json::jobject::parse(content);

	*out = jsonResult;
	return true;
}

void JSONWrite(json::jobject obj, std::string filename)
{
	std::string exportedJSON = obj.pretty();
	std::ofstream fs(filename, std::ios::out);
	fs << exportedJSON;
	fs.close();
}