#pragma once
#include "objects.h"
#include "attribute_struct.h"
#include "mapData_struct.h"
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
		{
			variables.push_back(kernel->variables[i]);
			variables[i].Generate(true);
		}
		for (int i = 0; i < kernel->PARAM_COUNT; i++)
		{
			parameters.push_back(kernel->parameters[i]);
			parameters[i].Generate(true);
		}
		for (int i = 0; i < kernel->MAP_COUNT; i++)
			mapDatas.push_back(kernel->mapDatas[i]);

		VAR_COUNT = kernel->VAR_COUNT;
		PARAM_COUNT = kernel->PARAM_COUNT;
		MAP_COUNT = kernel->MAP_COUNT;
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
			variables[i].ClearValues(); variables[i].Generate(false);
		}

		for (int i = 0; i < PARAM_COUNT; i++)
		{
			parameters[i].CalcStepCount();
			parameters[i].CalcStep();
			//if (!CUDA_kernel.parameters[i].DoValuesExist()) CUDA_kernel.parameters[i].Generate();
			parameters[i].ClearValues(); parameters[i].Generate(false);
		}
	}

	void AssessMapAttributes()
	{
		int varAttribute1 = -1;
		int varAttribute2 = -1;
		bool tooManyVarAttributes = false;
		MapDimensionType varType1 = VARIABLE;
		MapDimensionType varType2 = VARIABLE;

		for (int i = 0; i < VAR_COUNT; i++)
		{
			if (variables[i].TrueStepCount() > 1)
			{
				if (varAttribute1 == -1)
				{
					varAttribute1 = i;
					varType1 = VARIABLE;
				}
				else if (varAttribute2 == -1)
				{
					varAttribute2 = i;
					varType2 = VARIABLE;
				}
				else tooManyVarAttributes = true;
			}
		}

		for (int i = 0; i < PARAM_COUNT; i++)
		{
			if (parameters[i].TrueStepCount() > 1)
			{
				if (varAttribute1 == -1)
				{
					varAttribute1 = i;
					varType1 = PARAMETER;
				}
				else if (varAttribute2 == -1)
				{
					varAttribute2 = i;
					varType2 = PARAMETER;
				}
				else tooManyVarAttributes = true;
			}
		}

		for (int i = 0; i < MAP_COUNT; i++) mapDatas[i].toCompute = false;

		if (!tooManyVarAttributes && varAttribute1 > -1)
		{
			if (varAttribute2 > -1)
			{
				for (int i = 0; i < MAP_COUNT; i++)
				{
					mapDatas[i].indexX = varAttribute1;
					mapDatas[i].indexY = varAttribute2;
					mapDatas[i].typeX = varType1;
					mapDatas[i].typeY = varType2;
					mapDatas[i].toCompute = true;
				}
			}
			else
			{
				// TODO: if step
			}
		}
	}

	void MapsSetSizes()
	{
		unsigned long int mapsSize = 0;

		for (int i = 0; i < MAP_COUNT; i++)
		{
			if (!mapDatas[i].toCompute) continue;

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
				mapDatas[i].xSize = steps;
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
				mapDatas[i].ySize = steps;
				break;
			}

			mapDatas[i].offset = mapsSize;
			mapsSize += mapDatas[i].xSize * mapDatas[i].ySize;
		}
	}
};

struct MarshalledKernel : Kernel
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
};