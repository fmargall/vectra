#include <gtest/gtest.h>

#include <vectra/vectra.hpp>

TEST(BackendNoneFloat, StaticProperties)
{
	using Backend = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;
	
	static_assert(Backend::width() == 1);
	static_assert(std::is_same_v<Backend::type, float>);
	
	EXPECT_EQ(Backend::width(), 1);
	EXPECT_EQ(Backend::alignment(), alignof(float));
}

TEST(BackendNoneFloat, Arithmetic)
{
	using Backend = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;

	float a = 2.f;
	float b = 3.f;

	EXPECT_FLOAT_EQ(Backend::add(a, b),  5.f);
	EXPECT_FLOAT_EQ(Backend::sub(a, b), -1.f);
	EXPECT_FLOAT_EQ(Backend::mul(a, b),  6.f);
	EXPECT_FLOAT_EQ(Backend::div(a, b), 2.f / 3.f);
}

TEST(BackendNone, MathFunctions)
{
	using Backend = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;

	EXPECT_FLOAT_EQ(Backend::sqrt(4.f), 2.f);
	EXPECT_FLOAT_EQ(Backend::sin (0.f), 0.f);
	EXPECT_FLOAT_EQ(Backend::cos (0.f), 1.f);
}

TEST(BackendNoneFloat, IsTrivial)
{
	using Backend = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;

	static_assert(std::is_trivially_copyable_v<Backend::type>);
	SUCCEED();
}