#pragma once
#include "objects.h"
#include "attribute_struct.h"
//#include "main.h"

struct Kernel
{
public:
	std::string name;
	int steps;
	numb stepSize;
	bool executeOnLaunch;

	std::vector<Attribute> variables;
	std::vector<Attribute> parameters;
	std::vector<MapData> mapDatas;

	int VAR_COUNT;
	int PARAM_COUNT;
	int MAP_COUNT;

	void calcAttributeCounts()
	{
		VAR_COUNT = (int)variables.size();
		PARAM_COUNT = (int)parameters.size();
		MAP_COUNT = (int)mapDatas.size();
	}

	void CopyFrom(Kernel* kernel)
	{
		name = kernel->name;
		steps = kernel->steps;
		stepSize = kernel->stepSize;
		executeOnLaunch = kernel->executeOnLaunch;

		for (Attribute& v : variables)	v.ClearValues(); variables.clear();
		for (Attribute& p : parameters)	p.ClearValues(); parameters.clear();
		mapDatas.clear();

		for (int i = 0; i < kernel->VAR_COUNT; i++)
			variables.push_back(kernel->variables[i]);
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
			parameters.push_back(kernel->parameters[i]);
		for (int i = 0; i < kernel->MAP_COUNT; i++)
			mapDatas.push_back(kernel->mapDatas[i]);

		VAR_COUNT = kernel->VAR_COUNT;
		PARAM_COUNT = kernel->PARAM_COUNT;
		MAP_COUNT = kernel->MAP_COUNT;
	}
};

struct MarshalledKernel
{
public:
	int steps;
	numb stepSize;

	Attribute variables[MAX_ATTRIBUTES];
	Attribute parameters[MAX_ATTRIBUTES];
	MapData mapDatas[MAX_MAPS];

	int VAR_COUNT;
	int PARAM_COUNT;
	int MAP_COUNT;

	void CopyFrom(Kernel* kernel)
	{
		steps = kernel->steps;
		stepSize = kernel->stepSize;

		for (int i = 0; i < kernel->VAR_COUNT; i++)
			variables[i] = kernel->variables[i];
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
			parameters[i] = kernel->parameters[i];
		for (int i = 0; i < kernel->MAP_COUNT; i++)
			mapDatas[i] = kernel->mapDatas[i];

		VAR_COUNT = kernel->VAR_COUNT;
		PARAM_COUNT = kernel->PARAM_COUNT;
		MAP_COUNT = kernel->MAP_COUNT;
	}

	void CopyTo(Kernel* kernel)
	{
		kernel->steps = steps;
		kernel->stepSize = stepSize;

		for (Attribute& v : kernel->variables)	v.ClearValues(); kernel->variables.clear();
		for (Attribute& p : kernel->parameters)	p.ClearValues(); kernel->parameters.clear();
		kernel->mapDatas.clear();

		for (int i = 0; i < kernel->VAR_COUNT; i++)
			kernel->variables.push_back(variables[i]);
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
			kernel->parameters.push_back(parameters[i]);
		for (int i = 0; i < kernel->MAP_COUNT; i++)
			kernel->mapDatas.push_back(mapDatas[i]);

		kernel->VAR_COUNT = VAR_COUNT;
		kernel->PARAM_COUNT = PARAM_COUNT;
		kernel->MAP_COUNT = MAP_COUNT;
	}

	void MapsSetSizes()
	{
		/*unsigned long int mapsSize = 0;

		for (int i = 0; i < MAP_COUNT; i++)
		{
			int index = mapDatas[i].indexX;
			switch (mapDatas[i].typeX)
			{
			case VARIABLE:
				mapDatas[i].xSize = variables[index].stepCount;
				break;
			case PARAMETER:
				mapDatas[i].xSize = parameters[index].stepCount;
				break;
			case STEP:
				mapDatas[i].xSize = KERNEL.steps;
				break;
			}

			index = mapDatas[i].indexY;
			switch (mapDatas[i].typeY)
			{
			case VARIABLE:
				mapDatas[i].ySize = variables[index].stepCount;
				break;
			case PARAMETER:
				mapDatas[i].ySize = parameters[index].stepCount;
				break;
			case STEP:
				mapDatas[i].ySize = KERNEL.steps;
				break;
			}

			mapDatas[i].offset = mapsSize;
			mapsSize += mapDatas[i].xSize * mapDatas[i].ySize;
		}*/
	}
};