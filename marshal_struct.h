#pragma once
#include "kernel_struct.h"

#define MAX_MAP_SETTINGS 256

struct Marshal
{
	// Data struct that is pre-filled on the host, used and filled in the kernel and gets exported like PostRanging
	// But you CAN'T use it on the device because of vectors etc.
	// So instead we have an array-based dummy Kernel struct for marshalling, called MarshalledKernel
	MarshalledKernel kernel;

	// Trajectory data
	numb* trajectory;

	// Parameter values for each variation
	numb* parameterVariations;

	// Attribute step indices for each variation
	int* stepIndices;

	// Map data
	numb* maps;

	// Ranging attribute combinations count
	int totalVariations;	

	// Trajectory steps * variables = amount of numbers for one trajectory
	int variationSize;

	int mapsSize;

	// An array of map setting values, to be then acquired and employed by the kernel
	int mapSettings[MAX_MAP_SETTINGS];

	void CopyMetadataFrom(Marshal* marshal)
	{
		totalVariations = marshal->totalVariations;
		variationSize = marshal->variationSize;
		mapsSize = marshal->mapsSize;
	}
};