#pragma once
#include "numb.h"
#include <vector>
#include "../recurrenceCalculation.h"

struct RecurrenceProperties
{
	double epsilon;
	int decimation;
	int size;
	int steps;
	double* valueBuffer;
	unsigned char* pixelBuffer;
	void* texture;
	double min, max;
	RQA rqa;

	bool optimRR;
	double targetRR;

	std::vector<RQA> rqaHistory;
	uint64_t rqaVariations;
	uint64_t rqaBuffers;
	std::vector<RQA> rqaDensityAvgHistory;
	std::vector<RQA> rqaDensityRPPHistory;

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

		optimRR = true;
		targetRR = 0.01;

		rqaVariations = 0;
		rqaBuffers = 0;
	}

#define i4(o) i * 4 + o
	void MapToImg(double* mapBuffer, unsigned char** dataBuffer, int width, int height, bool global, double min, double max, ImPlotColormap colormap)
	{
		numb v;
		uint64_t i;
		ImVec4 c;

		if (!global)
		{
			for (int y = 0; y < height; y++)
				for (int x = 0; x < width; x++)
				{
					i = y * width + x;
					v = mapBuffer[i];

					if (v > 0.0)
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
		else
		{
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
	}

	void Prepare(Computation* cmp)
	{
		steps = cmp->marshal.kernel.steps;
		size = steps / decimation;
		if (size == 0) return;

		if (valueBuffer != nullptr)
		{
			delete[] valueBuffer;
		}
		valueBuffer = new double[size * size];
	}

	void Calculate(Computation* cmp, uint64_t variation, std::vector<int> vars)
	{
		CalculateRecurrence(cmp->marshal.trajectory, vars, size, steps, decimation, valueBuffer, epsilon, cmp->marshal.kernel.VAR_COUNT, variation);
		rqa = RecurrenceRQA(valueBuffer, size, 2, 2);
	}

	void CalculateGlobal(Computation* cmp, uint64_t variation, std::vector<int> vars)
	{
		CalculateRecurrenceGlobal(cmp->marshal.trajectory, vars, size, steps, decimation, valueBuffer, cmp->marshal.kernel.VAR_COUNT, variation);
	}

	void MakeRQA()
	{
		rqa = RecurrenceRQA(valueBuffer, size, 2, 2);
	}

	void MakeImage(double* values, bool global, double max)
	{
		if (pixelBuffer != nullptr)
		{
			delete[] pixelBuffer;
		}
		pixelBuffer = new unsigned char[size * size * 4];
		MapToImg(values, &pixelBuffer, size, size, global, 0.0, 1.0, ImPlotColormap_Greys);
	}

	double FindEpsilon(Computation* cmp, uint64_t variation, std::vector<int> vars, double rr)
	{
		CalculateRecurrenceGlobal(cmp->marshal.trajectory, vars, size, steps, decimation, valueBuffer, cmp->marshal.kernel.VAR_COUNT, variation);

		uint64_t targetRRindex = rr * size * size;
		double* sortedValues = new double[size * size];
		memcpy(sortedValues, valueBuffer, sizeof(double) * size * size);
		std::sort(sortedValues, sortedValues + size * size - 1);
		double epsilon = sortedValues[targetRRindex];
		delete[] sortedValues;

		return epsilon;
	}

	void SaveRQAToHistory(uint64_t variations)
	{
		rqaHistory.push_back(rqa);
		if (rqaBuffers == 0) rqaVariations = variations;
	}

	void ClearRQAHistory()
	{
		rqaHistory.clear();
		rqaVariations = 0;
		rqaBuffers = 0;
	}
};