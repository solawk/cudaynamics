#include "splitString.h"

std::vector<std::string> splitString(std::string str)
{
	// string split by Arafat Hasan
	// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
	size_t pos_start = 0, pos_end, delim_len = 1;
	std::string token;
	std::vector<std::string> data;
	while ((pos_end = str.find(" ", pos_start)) != std::string::npos)
	{
		token = str.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		data.push_back(token);
	}
	data.push_back(str.substr(pos_start));

	return data;
}