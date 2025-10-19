#pragma once
#include <vector>
#include <string>

// Display constraints for showing a parameter (e.g. only show a "symmetry" parameter for symmetrical methods)
// Each parameter has this structure (so we don't tap into the attribute struct), but only the constrained parameters have actual predicates there
struct Constraint
{
	bool hasConstraints;
	int count;
	std::vector<std::string> lhs;
	std::vector<std::string> rhs;

	void Clear()
	{
		hasConstraints = false;
		count = 0;
		lhs.clear();
		rhs.clear();
	}
};