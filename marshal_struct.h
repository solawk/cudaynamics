#pragma once
#include "kernel_struct.h"

struct Marshal
{
	// Data struct that is pre-filled on the host, used and filled in the kernel and gets exported like PostRanging
	// But you CAN'T use it on the device because of vectors etc.
	// So instead we have an array-based dummy Kernel struct for marshalling, called MarshalledKernel
	MarshalledKernel kernel;

	// Trajectory data
	numb* trajectory;

	// Variable values at the start of the trajectory (cheaper than memcpy'ing the entire trajectory to the device)
	numb* variableInits;

	// Parameter values for each variation
	numb* parameterVariations;

	// Attribute step indices for each variation
	int* stepIndices;

	// Map (analysis index) data
	numb* maps;

	// Analysis index values delta (for delta and decay plots)
	numb* indecesDelta;
	bool indecesDeltaExists;

	// Ranging attribute combinations count
	int totalVariations;	

	// Trajectory steps * variables = amount of numbers for one trajectory
	int variationSize;

	//int mapCount;
	unsigned int totalMapValuesPerVariation; // Previously mapCount meant total amount of map values per variation

	void CopyMetadataFrom(Marshal* marshal)
	{
		totalVariations = marshal->totalVariations;
		variationSize = marshal->variationSize;
		//mapCount = marshal->mapCount;
		totalMapValuesPerVariation = marshal->totalMapValuesPerVariation;
	}
};