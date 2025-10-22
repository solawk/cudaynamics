#include "map_img.h"

#define i4(o) i * 4 + o

void MapToImg(numb* mapBuffer, unsigned char** dataBuffer, int width, int height, numb min, numb max, ImPlotColormap colormap)
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
				c = ImPlot::SampleColormap(0.0f, colormap);
			}
			else if (v >= max)
			{
				c = ImPlot::SampleColormap(1.0f, colormap);
			}
			else
			{
				c = ImPlot::SampleColormap((float)((v - min) / (max - min)), colormap);
			}

			(*dataBuffer)[i4(0)] = (int)(c.x * 255);
			(*dataBuffer)[i4(1)] = (int)(c.y * 255);
			(*dataBuffer)[i4(2)] = (int)(c.z * 255);
			(*dataBuffer)[i4(3)] = (int)(c.w * 255);
		}
}

void MultichannelMapToImg(HeatmapProperties* heatmap, unsigned char** dataBuffer, int width, int height, bool ch0, bool ch1, bool ch2)
{
	numb v[3];
	ImVec4 c;
	int i;
	numb min, max;
	bool channelExists[3]{ heatmap->channel[0].valueBuffer != nullptr && ch0, heatmap->channel[1].valueBuffer != nullptr && ch1, heatmap->channel[2].valueBuffer != nullptr && ch2 };

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			i = y * width + x;
			for (int c = 0; c < 3; c++)
			{
				v[c] = channelExists[c] ? heatmap->channel[c].valueBuffer[i] : (numb)0.0;

				if (isnan(v[c]) || isinf(v[c]) || !channelExists[c])
				{
					(*dataBuffer)[i4(c)] = 0;
					continue;
				}

				min = heatmap->channel[c].heatmapMin;
				max = heatmap->channel[c].heatmapMax;

				if (v[c] <= min)
					(*dataBuffer)[i4(c)] = 0;
				else if (v[c] >= max)
					(*dataBuffer)[i4(c)] = 255;
				else
					(*dataBuffer)[i4(c)] = (char)(255.0f * (v[c] - min) / (max - min));
			}

			if (!channelExists[0] && !channelExists[1] && !channelExists[2])
				(*dataBuffer)[i4(3)] = 0;
			else
				(*dataBuffer)[i4(3)] = 255;
		}
}