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
	int sizePeaks;
	double* valueBuffer;
	unsigned char* pixelBuffer;
	unsigned char* pixelBufferPeaks;
	void* texture;
	void* texturePeaks;
	double min, max;
	RQA rqa;

	bool optimRR;
	double targetRR;

	double sigma;
	int variationToPlot;
	bool onlyPlot;

	std::vector<RQA> rqaHistory;
	std::vector<RQA> rqaPeaksHistory;
	uint64_t rqaVariations;
	uint64_t rqaBuffers;
	std::vector<RQA> rqaDensityAvgHistory;
	std::vector<RQA> rqaDensityRPPHistory;

	int peakVicinitySteps;
	bool chooseVicinityToRatio;
	float targetVicinityRatio;

	std::vector<double> histogram;

	// Peak finder
	double peakThreshold;    //minimum value of peak that can be found in peak finder, -inf by default	
	double maxAllowedValue;    //the maximum value allowed before peak finder deems system dispersive
	double epsFXP;  //eps area used in checking if system is a fixed point
	int analysedVariable; //variable of which peak finder analyses trajectory
	double timeFractionFXP;    //fraction of the trajectory that the system need to b e fixed point for peak finder to deem it a fixed point

	int windowsPerBuffer;

	RecurrenceProperties()
	{
		epsilon = 0.1;
		decimation = 1;
		size = 0;
		sizePeaks = 0;
		steps = 0;
		min = 0.0, max = 1.0;

		peakVicinitySteps = 0;
		chooseVicinityToRatio = true;
		targetVicinityRatio = 0.1f;

		valueBuffer = nullptr;
		pixelBuffer = nullptr;
		pixelBufferPeaks = nullptr;
		texture = nullptr;
		texturePeaks = nullptr;

		optimRR = true;
		targetRR = 0.01;

		sigma = 1.0;
		variationToPlot = 0;
		onlyPlot = false;

		rqaVariations = 0;
		rqaBuffers = 0;

		peakThreshold = -INFINITY;    //minimum value of peak that can be found in peak finder, -inf by default	
		maxAllowedValue = 1e4;    //the maximum value allowed before peak finder deems system dispersive
		epsFXP = 0.001;  //eps area used in checking if system is a fixed point
		analysedVariable = 0; //variable of which peak finder analyses trajectory
		timeFractionFXP = 0.05;    //fraction of the trajectory that the system need to b e fixed point for peak finder to deem it a fixed point

		windowsPerBuffer = 1;
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

	void MakeRQA(bool isPeaks)
	{
		if (!isPeaks)
			rqa = RecurrenceRQA(valueBuffer, size, 2, 2);
		else
			rqa = RecurrenceRQA(valueBuffer, sizePeaks, 2, 2);
	}

	void Calculate(Computation* cmp, Computation* nextcmp, uint64_t variation, std::vector<int> vars, int windows, int windowIndex)
	{
		CalculateRecurrence(cmp->marshal.trajectory, nextcmp->marshal.trajectory, windows, windowIndex, 
			vars, size, steps, decimation, valueBuffer, epsilon, cmp->marshal.kernel.VAR_COUNT, variation, false);
		MakeRQA(false);
	}

	void CalculatePeaks(Computation* cmp, Computation* nextcmp, uint64_t variation, std::vector<int> vars, std::vector<int>& peaks1, std::vector<int>& peaks2, int windows, int windowIndex)
	{
		sizePeaks = SpecificWindowSize((int)peaks1.size(), (int)peaks2.size(), (double)windowIndex / windows);
		CalculateRecurrenceSpecific(cmp->marshal.trajectory, nextcmp->marshal.trajectory, windows, windowIndex,
			vars, peaks1, peaks2, sizePeaks, decimation, valueBuffer, epsilon, cmp->marshal.kernel.VAR_COUNT, variation, false);
		MakeRQA(true);
	}

	void CalculateGlobal(Computation* cmp, Computation* nextcmp, uint64_t variation, std::vector<int> vars, int windows, int windowIndex)
	{
		CalculateRecurrence(cmp->marshal.trajectory, nextcmp->marshal.trajectory, windows, windowIndex, 
			vars, size, steps, decimation, valueBuffer, 0.0, cmp->marshal.kernel.VAR_COUNT, variation, true);
	}

	void MakeImage(double* values, bool global, double max, bool isPeaks)
	{
		if (!isPeaks)
		{
			if (pixelBuffer != nullptr)
			{
				delete[] pixelBuffer;
			}
			pixelBuffer = new unsigned char[size * size * 4];
			MapToImg(values, &pixelBuffer, size, size, global, 0.0, max, ImPlotColormap_Greys);
		}
		else
		{
			if (pixelBufferPeaks != nullptr)
			{
				delete[] pixelBufferPeaks;
			}
			pixelBufferPeaks = new unsigned char[sizePeaks * sizePeaks * 4];
			MapToImg(values, &pixelBufferPeaks, sizePeaks, sizePeaks, global, 0.0, max, ImPlotColormap_Greys);
		}
	}

	double FindEpsilon(Computation* cmp, Computation* nextcmp, uint64_t variation, std::vector<int> vars, double rr, int windows, int windowIndex)
	{
		CalculateRecurrence(cmp->marshal.trajectory, nextcmp->marshal.trajectory, windows, windowIndex, 
			vars, size, steps, decimation, valueBuffer, 0.0, cmp->marshal.kernel.VAR_COUNT, variation, true);

		// Main diagonal will always be zero, so we skip "size" of first elements in the sorted array
		uint64_t targetRRindex = rr * (size * size - 1);
		double* sortedValues = new double[size * size];
		memcpy(sortedValues, valueBuffer, sizeof(double) * size * size);
		std::sort(sortedValues, sortedValues + size * size - 1);
		double epsilon = sortedValues[size + targetRRindex];
		delete[] sortedValues;

		return epsilon;
	}

	double FindEpsilonPeaks(Computation* cmp, Computation* nextcmp, uint64_t variation, std::vector<int> vars, double rr, std::vector<int>& peaks1, std::vector<int>& peaks2,
		int windows, int windowIndex)
	{
		CalculateRecurrenceSpecific(cmp->marshal.trajectory, nextcmp->marshal.trajectory, windows, windowIndex, 
			vars, peaks1, peaks2, steps, decimation, valueBuffer, 0.0, cmp->marshal.kernel.VAR_COUNT, variation, true);

		int windowSize = SpecificWindowSize((int)peaks1.size(), (int)peaks2.size(), (double)windowIndex / windows);

		uint64_t targetRRindex = rr * windowSize * (windowSize - 1);
		double* sortedValues = new double[windowSize * windowSize];
		memcpy(sortedValues, valueBuffer, sizeof(double) * windowSize * windowSize);
		std::sort(sortedValues, sortedValues + windowSize * windowSize - 1);
		double epsilon = sortedValues[windowSize + targetRRindex];
		delete[] sortedValues;

		return epsilon;
	}

	void SaveRQAToHistory(uint64_t variations)
	{
		rqaHistory.push_back(rqa);
		if (rqaBuffers == 0) rqaVariations = variations;
	}

	void SaveRQAToHistoryPeaks(uint64_t variations)
	{
		rqaPeaksHistory.push_back(rqa);
		if (rqaBuffers == 0) rqaVariations = variations;
	}

	void ClearRQAHistory()
	{
		rqaHistory.clear();
		rqaPeaksHistory.clear();
		rqaVariations = 0;
		rqaBuffers = 0;
	}

	int SpecificWindowSize(int size1, int size2, float t)
	{
		return (int)roundf(size1 + t * (size2 - size1));
	}

	std::vector<int> PeakFinder(Computation* cmp, uint64_t variation)
	{
		std::vector<int> peakTimes;

		bool returnNan = false, returnZero = false, WritingData = true; //flags for if the system is dispersive, is a fixed point or if peakfinder has filled buffer and wont write any new peaks

		numb tempPeakAmp = 0, tempPeakTime = 0; bool tempPeakFound = false; // used in case if peak finder finds a horizontal line of equal values and doesnt know if there is a peak there until the line ends, while the line is being analysed the first value of the line is save into tempPeakAmp and tempPeakTime
		bool firstpeakreached = false; // flag for the first peak in trajectory which we dont save into data and use just for its interval with the next peak
		int temppeakindex;     // used to save the time of last peak in trajectory
		int fixedPointCount = 0;    // used to count how many continuous values in trajectory fulfiil the epsFXP requirement

		// NEW
		int steps = cmp->marshal.kernel.steps;
		numb* trajectory = cmp->marshal.trajectory + steps * variation;
		numb stepSize = cmp->marshal.kernel.GetStepSize();
		int varCount = cmp->marshal.kernel.VAR_COUNT;
		int fixedPointMaxCount = round(steps * timeFractionFXP);   //amount of steps in trajectory that the system need to be fixed point for peak finder to deem it a fixed point

		//  Peak finder
		for (int s = 1; s < steps - 1; s++)
		{
			numb prev = trajectory[analysedVariable + varCount * s - varCount];
			numb curr = trajectory[analysedVariable + varCount * s];
			numb next = trajectory[analysedVariable + varCount * s + varCount];

			if (abs(next - curr) / stepSize < epsFXP) // check the derivative for fixed point requirement
			{
				fixedPointCount++;
				if (fixedPointCount > fixedPointMaxCount) returnZero = true;
			}
			else
				fixedPointCount = 0;

			if (abs(curr) > maxAllowedValue) //check if value is too big to be a dispersive system
				returnNan = true;
			else if (curr > peakThreshold) 
			{
				if (curr > prev && curr > next)     //peak found
				{
					tempPeakFound = false;
					if (firstpeakreached == false)
					{
						firstpeakreached = true;
						temppeakindex = s;
					}
					else
					{
						if (WritingData) 
						{
							peakTimes.push_back(s);
						}

						temppeakindex = s;
					}
				}
				else if (curr == next && curr > prev) 
				{ // found a possible peak solved as a line by finiteDifferenceScheme
					tempPeakFound = true; tempPeakAmp = curr; tempPeakTime = s;
				}
				else if (curr < next) 
				{    // case in which the line was likely not a peak
					tempPeakFound = false;
				}
				else if (curr > next && tempPeakFound) 
				{    // the line value is larger than values before and after the line, which means first value of line is taken as peak
					if (firstpeakreached) 
					{
						if (WritingData)
						{
							peakTimes.push_back(s);
						}

						temppeakindex = tempPeakTime;
						tempPeakFound = false;
					}
					else 
					{
						firstpeakreached = true;
						temppeakindex = tempPeakTime;
						tempPeakFound = false;
					}
				}
			}
		}

		//for (int i = 0; i < peakTimes.size(); i++) printf("non %f\n", (float)peakTimes[i]);
		return peakTimes;
	}

	std::vector<int> BroadenPeaks(std::vector<int>& peaks)
	{
		if (peakVicinitySteps > 0)
		{
			std::vector<int> peaksWithVicinity;
			for (int i = 0; i < peaks.size(); i++)
			{
				for (int v = -peakVicinitySteps; v <= peakVicinitySteps; v++)
				{
					if (peaks[i] + v >= 0 && peaks[i] + v < steps)
						peaksWithVicinity.push_back(peaks[i] + v);
				}
			}
			std::sort(begin(peaksWithVicinity), end(peaksWithVicinity));
			for (int i = 0; i < (int)peaksWithVicinity.size() - 1; i++)
			{
				if (peaksWithVicinity[i] == peaksWithVicinity[i + 1])
				{
					peaksWithVicinity.erase(peaksWithVicinity.begin() + i + 1);
					i--;
				}
			}

			//for (int i = 0; i < peaksWithVicinity.size(); i++) printf("vic %f\n", (float)peaksWithVicinity[i]);
			return peaksWithVicinity;
		}

		return peaks;
	}

	void OptimizePeakVicinity(std::vector<int>& peaks1, std::vector<int>& peaks2)
	{
		int peakCount = (int)peaks1.size() + (int)peaks2.size();
		float targetSteps = (float)steps * targetVicinityRatio;
		float stepsPerPeakF = (targetSteps * 2.0f) / peakCount; // Searching for average vicinity between current and next buffer's peaks
		int stepsPerPeak;
		if ((int)stepsPerPeakF % 2 == 1)
			stepsPerPeak = (int)stepsPerPeakF;
		else
			stepsPerPeak = (int)stepsPerPeakF + 1;
		int vicinity = (stepsPerPeak - 1) / 2;
		if (vicinity < 0)
		{
			printf("Vicinity < 0!!! Setting to 0\n");
			vicinity = 0;
		}
		peakVicinitySteps = vicinity;
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
};