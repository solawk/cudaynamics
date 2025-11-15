#pragma once

// Ports to connect Indices to analysis functions
struct Port
{
	bool used;
	unsigned int offset;
	unsigned int size; // previously valueCount

	Port()
	{
		used = false;
		offset = 0;
		size = 1;
	}
};