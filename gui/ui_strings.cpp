#include "ui_strings.h"

std::string rangingTypes[] = { "Fixed", "Step", "Linear", "Random", "Normal", "Enum" };

std::string rangingDescriptions[] =
{
    "Single value",
    "Values from 'min' to 'max' (inclusive), separated by 'step'",
    "Specified amount of values between 'min' and 'max' (inclusive)",
    "Uniform random distribution of values between 'min' and 'max'",
    "Normal random distribution of values around 'mu' with standard deviation 'sigma'"
};

std::string plottypes[] = { "Variables time series", "3D Phase diagram", "2D Phase diagram", "Orbit diagram", "Heatmap", "RGB Heatmap", "Indices diagram", "Indices time series", "Decay plot" };

std::string variablexyz[] = { "x", "y", "z" };