#pragma once
#include "numb.h"
#include <vector>
#include "../recurrenceCalculation.h"

struct RecurrenceProperties
{
	numb epsilon;
	int decimation;
	int size;
	int steps;
	bool* valueBuffer;
	//numb* valueBuffer;
	unsigned char* pixelBuffer;
	void* texture;
	numb min, max;
	RQA rqa;

	std::vector<RQA> rqaHistory;

	RecurrenceProperties()
	{
		epsilon = 0.1;
		decimation = 1;
		size = 0;
		steps = 0;
		min = 0.0, max = 1.0;

		valueBuffer = nullptr;
		pixelBuffer = nullptr;
		texture = nullptr;
	}

#define i4(o) i * 4 + o
	void MapToImgWDecimationBinary(bool* mapBuffer, unsigned char** dataBuffer, int width, int height, int decX, int decY)
	{
		bool v;
		uint64_t i;

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				i = y * width + x;
				v = mapBuffer[i];

				if (v)
				{
					(*dataBuffer)[i4(0)] = 0;
					(*dataBuffer)[i4(1)] = 0;
					(*dataBuffer)[i4(2)] = 0;
					(*dataBuffer)[i4(3)] = 255;
				}
				else
				{

					(*dataBuffer)[i4(0)] = 255;
					(*dataBuffer)[i4(1)] = 255;
					(*dataBuffer)[i4(2)] = 255;
					(*dataBuffer)[i4(3)] = 255;
				}
			}
	}
	void MapToImgAnalog(numb* mapBuffer, unsigned char** dataBuffer, int width, int height, numb min, numb max, ImPlotColormap colormap)
	{
		numb v;
		ImVec4 c;
		uint64_t i;

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				i = y * width + x;
				v = mapBuffer[i];

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

	void Calculate(Computation* cmp, uint64_t variation, std::vector<int> vars)
	{
		steps = cmp->marshal.kernel.steps;
		size = steps / decimation;
		if (size == 0) return;

		if (valueBuffer != nullptr)
		{
			delete[] valueBuffer;
		}
		valueBuffer = new bool[size * size];
		//valueBuffer = new numb[size * size];

		if (pixelBuffer != nullptr)
		{
			delete[] pixelBuffer;
		}
		pixelBuffer = new unsigned char[size * size * 4];

		// Recurrence calculation
		//for (int i = 0; i < size * size; i++) valueBuffer[i] = i % 2;
		// 
		CalculateRecurrence(cmp->marshal.trajectory, vars, size, steps, decimation, valueBuffer, epsilon, cmp->marshal.kernel.VAR_COUNT, variation);
		rqa = RecurrenceRQA(valueBuffer, size, 2, 2);
		MapToImgWDecimationBinary(valueBuffer, &pixelBuffer, size, size, decimation, decimation);

		//CalculateRecurrenceAnalog(cmp->marshal.trajectory, vars, size, steps, decimation, valueBuffer, cmp->marshal.kernel.VAR_COUNT, variation);
		//MapToImgAnalog(valueBuffer, &pixelBuffer, size, size, min, max, ImPlotColormap_Plasma);
	}

	void SaveRQAToHistory()
	{
		rqaHistory.push_back(rqa);
	}

	void ClearRQAHistory()
	{
		rqaHistory.clear();
	}
};