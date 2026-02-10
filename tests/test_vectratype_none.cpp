#include <gtest/gtest.h>

#include <vectra/vectra.hpp>

TEST(VectratypeNoneFloat, Arithmetic)
{
	using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;

    vct a(2.f);
    vct b(3.f);
	
    vct c = a + b;
    vct d = a * b;
    vct e = d - a;
    vct f = d / b;

    EXPECT_FLOAT_EQ(c.hsum(), 5.f);
    EXPECT_FLOAT_EQ(d.hsum(), 6.f);
    EXPECT_FLOAT_EQ(e.hsum(), 4.f);
    EXPECT_FLOAT_EQ(f.hsum(), 2.f);
}