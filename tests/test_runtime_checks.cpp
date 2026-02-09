#include <iostream>

#include <gtest/gtest.h>

#include <vectra/dispatch/runtime_checks.hpp>

TEST(RuntimeChecks, HighestLevel) {
	auto level = vectra::highestRuntimeSIMDLevel();

	// SIMD level check for current machine, at runtime
	std::cout << "[ Vectra   ] Highest detected SIMD level: " 
		      << level << std::endl;

	EXPECT_GE(static_cast<int>(level), 0);
}