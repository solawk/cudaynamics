#pragma once
#include "numb.h"
#include <Eigen/Dense>

constexpr size_t NumTDAMetrics = 7;

struct TDAMetrics
{
	double meanA, meanI;
	double sigmaA, sigmaI;
	double covAA, covAI, covII;
	double lambda1, lambda2;
	double effectiveArea, logArea;
	double elongation;
	double axisRatio, principalAngle;

	double metrics[NumTDAMetrics]{ 0.0 };

	TDAMetrics()
	{
		meanA = meanI = 0.0;
		sigmaA = sigmaI = 0.0;
		covAA = covAI = covII = 0.0;
		lambda1 = lambda2 = 0.0;
		effectiveArea = logArea = 0.0;
		elongation = 0.0;
		axisRatio = principalAngle = 0.0;
	}
};

struct TDAAnalysisResult
{
	std::vector<double> changeRate;

	int transitionIndex = 0;
};

struct TDAProperties
{
	// Peak finder
	double peakThreshold;    //minimum value of peak that can be found in peak finder, -inf by default	
	double maxAllowedValue;    //the maximum value allowed before peak finder deems system dispersive
	double epsFXP;  //eps area used in checking if system is a fixed point
	int analysedVariable; //variable of which peak finder analyses trajectory
	double timeFractionFXP;    //fraction of the trajectory that the system need to b e fixed point for peak finder to deem it a fixed point

	std::vector<int> peakTimes;
	std::vector<double> peakAmplitudes;
	std::vector<double> peakIntervals;

	std::vector<TDAMetrics> metrics;
	TDAAnalysisResult tdaar;

	TDAProperties()
	{
		peakThreshold = -INFINITY;
		maxAllowedValue = 1e4;
		epsFXP = 0.001;
		analysedVariable = 0;
		timeFractionFXP = 0.05;
	}

	void Clear()
	{
		metrics.clear();
	}

	TDAMetrics ComputeMetrics()
	{
		TDAMetrics metric;
		int peaks = (int)peakIntervals.size();

		// Mean

		double sumA = 0.0;
		double sumI = 0.0;

		for (int i = 0; i < peaks; i++)
		{
			sumA += peakAmplitudes[i];
			sumI += peakIntervals[i];
		}

		metric.meanA = sumA / peaks;
		metric.meanI = sumI / peaks;

		// Cov and sigma

		double sAA = 0.0;
		double sII = 0.0;
		double sAI = 0.0;

		for (int i = 0; i < peaks; i++)
		{
			double dA = peakAmplitudes[i] - metric.meanA;
			double dI = peakIntervals[i] - metric.meanI;

			sAA += dA * dA;
			sII += dI * dI;
			sAI += dA * dI;
		}

		double denom = peaks - 1.0;

		metric.covAA = sAA / denom;
		metric.covII = sII / denom;
		metric.covAI = sAI / denom;

		metric.sigmaA = std::sqrt(metric.covAA > 0.0 ? metric.covAA : 0.0);
		metric.sigmaI = std::sqrt(metric.covII > 0.0 ? metric.covII : 0.0);

		// Eigenvalues

		double a = metric.covAA;
		double b = metric.covAI;
		double c = metric.covII;

		double trace = a + c;
		double discriminant = (a - c) * (a - c) + 4.0 * b * b;
		if (discriminant < 0.0) discriminant = 0.0;

		double eig1 = 0.5 * (trace + discriminant);
		double eig2 = 0.5 * (trace - discriminant);

		if (eig1 < eig2)
		{
			std::swap(eig1, eig2);
		}

		metric.lambda1 = eig1 > 0.0 ? eig1 : 0.0;
		metric.lambda2 = eig2 > 0.0 ? eig2 : 0.0;

		// Area of the cloud

		const double regularizationEps = 1e-12;
		double detRegularized = (metric.covAA + regularizationEps) * (metric.covII + regularizationEps) - metric.covAI * metric.covAI;

		metric.effectiveArea = detRegularized > 0.0 ? sqrt(detRegularized) : 0.0;
		metric.logArea = std::log(metric.effectiveArea + regularizationEps);

		// Elongation

		double eigSum = metric.lambda1 + metric.lambda2;

		if (eigSum > regularizationEps)
		{
			metric.elongation = metric.lambda1 / eigSum;
		}
		else
		{
			metric.elongation = 0.5;
		}

		// Axis ratio

		if (metric.lambda2 > regularizationEps)
		{
			metric.axisRatio = sqrt(metric.lambda1 / metric.lambda2);
		}
		else
		{
			metric.axisRatio = INFINITY;
		}

		// Principal angle

		metric.principalAngle = abs(0.5 * std::atan2(2.0 * metric.covAI, metric.covAA - metric.covII));

		// Adding to array
		metric.metrics[0] = metric.meanA;
		metric.metrics[1] = metric.meanI;
		metric.metrics[2] = metric.sigmaA;
		metric.metrics[3] = metric.sigmaI;
		metric.metrics[4] = metric.lambda1;
		metric.metrics[5] = metric.logArea;
		metric.metrics[6] = metric.principalAngle;

		return metric;
	}

	TDAAnalysisResult AnalyzeTransientChaos(const std::vector<TDAMetrics>& windows)
	{
		TDAAnalysisResult result;

		const size_t N = windows.size();

		if (N < 2) return result;

		//----------------------------------------------------------
		// Build feature matrix
		//----------------------------------------------------------

		Eigen::MatrixXd X(N, NumTDAMetrics);

		for (size_t i = 0; i < N; ++i)
			for (size_t j = 0; j < NumTDAMetrics; ++j)
				X(i, j) = windows[i].metrics[j];

		//----------------------------------------------------------
		// Standardize each metric over all windows
		//----------------------------------------------------------

		for (size_t j = 0; j < NumTDAMetrics; ++j)
		{
			double mean = X.col(j).mean();

			double variance = 0.0;

			for (size_t i = 0; i < N; ++i)
			{
				double d = X(i, j) - mean;
				variance += d * d;
			}

			double sigma = std::sqrt(variance / (N - 1));

			if (sigma < 1e-12)
				sigma = 1.0;

			for (size_t i = 0; i < N; ++i)
				X(i, j) = (X(i, j) - mean) / sigma;
		}

		//----------------------------------------------------------
		// Change rate between consecutive windows
		//----------------------------------------------------------
		// Determining the transition point

		result.changeRate.resize(N);
		result.changeRate[0] = 0.0;
		double largestChange = 0.0;

		for (size_t i = 1; i < N; ++i)
		{
			Eigen::VectorXd delta = X.row(i) - X.row(i - 1);
			result.changeRate[i] = delta.norm();

			if (result.changeRate[i] > largestChange)
			{
				largestChange = result.changeRate[i];
				result.transitionIndex = (int)i;
			}
		}

		return result;
	}

	void PeakFinder(Computation* cmp, uint64_t variation)
	{
		peakTimes.clear();
		peakAmplitudes.clear();
		peakIntervals.clear();

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
							peakAmplitudes.push_back(curr);
							peakIntervals.push_back((s - temppeakindex) * stepSize);
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
							peakAmplitudes.push_back(tempPeakAmp);
							peakIntervals.push_back((tempPeakTime - temppeakindex) * stepSize);
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
	}
};