#include <gtest/gtest.h>

#include <vectra/dispatch/runtime_checks.hpp>

TEST(RuntimeChecks, HighestLevel) {
	auto level = vectra::highestRuntimeSIMDLevel();
	EXPECT_GE(static_cast<int>(level), 0);
}