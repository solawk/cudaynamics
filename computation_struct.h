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
	int index;
	bool isFirst;
	int bufferNo;

	void Clear()
	{
		if (marshal.trajectory != nullptr)				{ delete[] marshal.trajectory; marshal.trajectory = nullptr; }
		if (marshal.parameterVariations != nullptr)		{ delete[] marshal.parameterVariations; marshal.parameterVariations = nullptr; }
		if (marshal.stepIndices != nullptr)				{ delete[] marshal.stepIndices; marshal.stepIndices = nullptr; }
		//if (marshal.maps != nullptr)					{ delete[] marshal.maps; marshal.maps = nullptr; }
		if (marshal.maps2 != nullptr)					{ delete[] marshal.maps2; marshal.maps2 = nullptr; }

		ready = false;
		timeElapsed = 0.0f;
		isFirst = false;
		bufferNo = -1;
	}
};