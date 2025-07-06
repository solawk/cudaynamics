#pragma once
#include <future>
#include "objects.h"
#include "marshal_struct.h"

// Computation stores the marshal and metainformation about a CUDA launch
struct Computation
{
public:
	Marshal marshal;
	Marshal* otherMarshal;
	std::atomic_bool ready;
	std::future<int> future;
	int threads_per_block;
	float timeElapsed;
	int index; // 0 or 1

	bool isFirst;
	int bufferNo; // When computing starts, this holds the continuous index of the current buffer

	// Hi-res gizmos
	bool isHires; // True only for the hi-res computation
	unsigned long long variationsPerParallelization; // How many variations are launched in parallel during hi-res computation (in one execute)
	unsigned long long variationsInCurrentExecute; // Current variations count, can be less at the end of hi-res computation
	unsigned long long startVariationInCurrentExecute; // Index of the first variation in the execute
	int buffersPerVariation = 1; // How many times is one variation repeated (steps * buffersPerVariation equals total steps per variation)
	unsigned long long variationsFinished; // How many variations of marshal.totalVariations have finished hi-res-computing

	void Clear()
	{
		if (marshal.trajectory != nullptr)				{ delete[] marshal.trajectory;			marshal.trajectory = nullptr; }
		if (marshal.parameterVariations != nullptr)		{ delete[] marshal.parameterVariations;	marshal.parameterVariations = nullptr; }
		if (marshal.stepIndices != nullptr)				{ delete[] marshal.stepIndices;			marshal.stepIndices = nullptr; }
		if (marshal.maps != nullptr)					{ delete[] marshal.maps;				marshal.maps = nullptr; }

		ready = false;
		timeElapsed = 0.0f;

		isFirst = false;
		bufferNo = -1;
	}
};