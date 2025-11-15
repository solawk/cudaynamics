#pragma once
#include "objects.h"
#include "attribute_struct.h"
#include "mapData_struct.h"
#include "constraint.h"
#include "analysesSettings_struct.h"

// TODOs:
// We need to add a proper "none/fixed/variable" step
// We need to add decimation
// We need to add proper time/step calculation
// 1. Solution for steps: a flag marking a discrete step ("none"), a fixed step value ("fixed", parameter) or a buffer of step values ("variable", variable, since it can change in time).
//  - We need to treat a variable step like an actual variable, since we need to plot, copy it and all of that. The same goes with the parameter step.
//  - The path of least resistance is to add the step as a parameter or a variable. This restructures the kernel and reinitializes it.
//  - This will be the turning point: EVERY kernel will be initialized with a parameter or a variable step added to it automatically, depending on its type, making step innate.
// 3. How to mix time into this architecture? Since Time = Steps * StepSize, Steps = Time / StepSize.
//  - When step size is discrete, steps simply equal time.
//  - When step size is fixed, step counts can be different for different variations. We either resort to allocating memory for maximum amount of steps or automatically decimate the trajectories.
//  - When step size is variable, we can only set the maximum amount of steps, that we expect to not exhaust by a certain time. For clarity, we should export elapsed simulation time and used steps.
// 2. Decimation should work by only recording a portion of the steps. By having a target step count we decide if we record the step or not.
//  - We introduce local recording counters in a kernel, a float and an int. If there is no decimation, the float one incremented by 1.0 each step. Whenever floor(float counter) >= 1.0, a step is recorded.
//  - If there is decimation, it is incremented by DecimatedSteps/SimulationSteps. E.g., if 400 steps are decimated into 100, we increment the float counter by 0.25 each step, recording every fourth step.
//  - If decimation is irrational, e.g. 9 sim steps into 5: +Start, 5/9, +10/9, 15/9, +20/9, 25/9, +30/9, 35/9, +40/9, +45/9. The last step is always recorded.
//  - 11 steps into 3: +Start, 3/11, 6/11, 9/11, +12/11, 15/11, 18/11, 21/11, +24/11, 27/11, 30/11, +33/11
//  - 5 steps into 4: +Start, 4/5, +8/5, +12/5, +16/5, +20/5
// Roadmap:
// 1. Add automatic addition of step size as a parameter or a variable. Add a system setting for this and an option selection in the UI.
// 2. Add a macro to use the appropriate step size. TO DETERMINE: how should analyses use the variable step size
// 3. Add decimation.

struct Kernel
{
public:
	std::string name;

	int steps;
	int transientSteps;
	float time;
	float transientTime;
	bool usingTime;
	numb stepSize;

	StepType stepType;

	bool executeOnLaunch;
	float mapWeight;

	std::vector<Attribute> variables;
	std::vector<Attribute> parameters;
	std::vector<Constraint> constraints;

	AnalysesSettings analyses;
	
	//std::vector<MapData> mapDatas;

	int VAR_COUNT;
	int PARAM_COUNT;
	//int MAP_COUNT;

	// An array of map setting values
	//numb mapSettings[MAX_MAP_SETTINGS];

	void calcAttributeCounts()
	{
		VAR_COUNT = (int)variables.size();
		PARAM_COUNT = (int)parameters.size();
		//MAP_COUNT = (int)mapDatas.size();
	}

	void CopyFrom(Kernel* kernel)
	{
		name = kernel->name;
		steps = kernel->steps;
		transientSteps = kernel->transientSteps;
		time = kernel->time;
		transientTime = kernel->transientTime;
		usingTime = kernel->usingTime;
		stepSize = kernel->stepSize;
		executeOnLaunch = kernel->executeOnLaunch;
		mapWeight = kernel->mapWeight;
		stepType = kernel->stepType;

		for (Attribute& v : variables)	v.ClearValues(); variables.clear();
		for (Attribute& p : parameters)	p.ClearValues(); parameters.clear();
		constraints.clear();

		for (int i = 0; i < kernel->VAR_COUNT; i++)
		{
			variables.push_back(kernel->variables[i]);
			variables[i].Generate(true);
		}
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
		{
			parameters.push_back(kernel->parameters[i]);
			parameters[i].Generate(true);
		}
		for (int i = 0; i < kernel->constraints.size(); i++)
			constraints.push_back(kernel->constraints[i]);

		VAR_COUNT = kernel->VAR_COUNT;
		PARAM_COUNT = kernel->PARAM_COUNT;
		
		analyses = kernel->analyses;

		//for (int i = 0; i < MAX_MAP_SETTINGS; i++) mapSettings[i] = kernel->mapSettings[i];
	}

	void CopyParameterValuesFrom(Kernel* kernel)
	{
		for (Attribute& p : parameters)	p.ClearValues(); parameters.clear();

		for (int i = 0; i < kernel->PARAM_COUNT; i++)
		{
			parameters.push_back(kernel->parameters[i]);
			parameters[i].Generate(true);
		}

		PARAM_COUNT = kernel->PARAM_COUNT;
	}

	void PrepareAttributes()
	{
		for (int i = 0; i < VAR_COUNT; i++)
		{
			variables[i].CalcStepCount();
			variables[i].CalcStep();
			//if (!CUDA_kernel.variables[i].DoValuesExist()) CUDA_kernel.variables[i].Generate(); TODO
			variables[i].ClearValues();
			variables[i].Generate(false);
			if (variables[i].TrueStepCount() == 1) variables[i].selectedForMaps = false;
		}

		for (int i = 0; i < PARAM_COUNT; i++)
		{
			parameters[i].CalcStepCount();
			parameters[i].CalcStep();
			//if (!CUDA_kernel.parameters[i].DoValuesExist()) CUDA_kernel.parameters[i].Generate();
			parameters[i].ClearValues();
			parameters[i].Generate(false);
			if (parameters[i].TrueStepCount() == 1) parameters[i].selectedForMaps = false;
		}
	}

	void AssessMapAttributes()
	{
		/*for (int i = 0; i < MAP_COUNT; i++)
		{
			mapDatas[i].toCompute = mapDatas[i].userEnabled;
		}*/
	}
};

struct MarshalledKernel : Kernel
{
public:
	std::string name;

	int steps;
	int transientSteps;
	float time;
	float transientTime;
	bool usingTime;
	numb stepSize;

	StepType stepType;

	Attribute variables[MAX_ATTRIBUTES];
	Attribute parameters[MAX_ATTRIBUTES];
	MapData mapDatas[MAX_MAPS]; // TODO: remove and remove from systems

	int VAR_COUNT;
	int PARAM_COUNT;
	int MAP_COUNT;

	AnalysesSettings analyses;

	//numb mapSettings[MAX_MAP_SETTINGS];

	void CopyFrom(Kernel* kernel)
	{
		name = kernel->name;
		steps = kernel->steps;
		transientSteps = kernel->transientSteps;
		time = kernel->time;
		transientTime = kernel->transientTime;
		usingTime = kernel->usingTime;
		stepSize = kernel->stepSize;
		mapWeight = kernel->mapWeight;
		stepType = kernel->stepType;

		for (int i = 0; i < kernel->VAR_COUNT; i++)
			variables[i] = kernel->variables[i];
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
			parameters[i] = kernel->parameters[i];

		VAR_COUNT = kernel->VAR_COUNT;
		PARAM_COUNT = kernel->PARAM_COUNT;

		analyses = kernel->analyses;

		//memcpy(mapSettings, kernel->mapSettings, MAX_MAP_SETTINGS * sizeof(numb));
	}

	void CopyTo(Kernel* kernel)
	{
		kernel->name = name;
		kernel->steps = steps;
		kernel->transientSteps = transientSteps;
		kernel->stepSize = stepSize;
		kernel->mapWeight = mapWeight;
		kernel->stepType = stepType;

		for (Attribute& v : kernel->variables)	v.ClearValues(); kernel->variables.clear();
		for (Attribute& p : kernel->parameters)	p.ClearValues(); kernel->parameters.clear();

		for (int i = 0; i < kernel->VAR_COUNT; i++)
			kernel->variables.push_back(variables[i]);
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
			kernel->parameters.push_back(parameters[i]);

		kernel->VAR_COUNT = VAR_COUNT;
		kernel->PARAM_COUNT = PARAM_COUNT;

		kernel->analyses = analyses;
	}
};