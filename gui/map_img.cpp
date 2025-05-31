#include "map_img.h"

#define i4(o) i * 4 + o

void MapToImg(numb* mapBuffer, unsigned char** dataBuffer, int width, int height, numb min, numb max)
{
	numb v;
	ImVec4 c;
	int i;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			i = y * width + x;
			v = mapBuffer[i];

			if (isnan(v) || isinf(v))
			{
				(*dataBuffer)[i4(0)] = 0;
				(*dataBuffer)[i4(1)] = 0;
				(*dataBuffer)[i4(2)] = 0;
				(*dataBuffer)[i4(3)] = 0;
				continue;
			}

			if (v <= min)
			{
				c = ImPlot::SampleColormap(0.0f, ImPlotColormap_Jet);
			}
			else if (v >= max)
			{
				c = ImPlot::SampleColormap(1.0f, ImPlotColormap_Jet);
			}
			else
			{
				c = ImPlot::SampleColormap((v - min) / (max - min), ImPlotColormap_Jet);
			}

			(*dataBuffer)[i4(0)] = (int)(c.x * 255);
			(*dataBuffer)[i4(1)] = (int)(c.y * 255);
			(*dataBuffer)[i4(2)] = (int)(c.z * 255);
			(*dataBuffer)[i4(3)] = (int)(c.w * 255);
		}
}