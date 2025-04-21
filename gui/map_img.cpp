#include "map_img.h"

#define i4(o) i * 4 + o

ImVec4 KeysJet[] = {	ImVec4(0.5f, 0.0f, 0.5f, 1.0f), ImVec4(0.0f, 0.0f, 1.0f, 1.0f),
						ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
						ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
						ImVec4(1.0f, 0.5f, 0.0f, 1.0f), ImVec4(1.0f, 0.0f, 0.0f, 1.0f) };

ImVec4 vec4lerp(ImVec4 v0, ImVec4 v1, float t)
{
	return ImVec4(v0.x + (v1.x - v0.x) * t, v0.y + (v1.y - v0.y) * t, v0.z + (v1.z - v0.z) * t, v0.w + (v1.w - v0.w) * t);
}

ImVec4 colorLerp(ImVec4* keys, int keyCount, float t)
{
	float keyRange = 1.0f / (keyCount - 1);
	int key0 = (int)(t / keyRange);
	int key1 = key0 + 1;
	float tBetween = (t - key0 * keyRange) / keyRange;
	return vec4lerp(keys[key0], keys[key1], tBetween);
}

void MapToImg(numb* mapBuffer, unsigned char** dataBuffer, int width, int height, numb min, numb max)
{
	numb v;
	ImVec4 c;

	for (int i = 0; i < width * height; i++)
	{
		v = mapBuffer[i];

		if (isnan(v) || isinf(v))
		{
			(*dataBuffer)[i4(0)] = 0;
			(*dataBuffer)[i4(1)] = 0;
			(*dataBuffer)[i4(2)] = 0;
			(*dataBuffer)[i4(3)] = 0;
			continue;
		}

		if (v <= min)
		{
			c = KeysJet[0];
		}
		else if (v >= max)
		{
			c = KeysJet[7];
		}
		else
		{
			c = colorLerp(KeysJet, 8, (v - min) / (max - min));
		}

		(*dataBuffer)[i4(0)] = (int)(c.x * 255);
		(*dataBuffer)[i4(1)] = (int)(c.y * 255);
		(*dataBuffer)[i4(2)] = (int)(c.z * 255);
		(*dataBuffer)[i4(3)] = (int)(c.w * 255);
	}
}