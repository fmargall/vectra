#include <gtest/gtest.h>

#include <vectra/types/pack.hpp>

TEST(PackNoneFloat, Arithmetic)
{
	using packet = vectra::Pack<float, vectra::SIMDLevel::None>;

    packet a(2.f);
    packet b(3.f);
	
    packet c = a + b;
    packet d = a * b;
    packet e = d - a;
    packet f = d / b;

    EXPECT_FLOAT_EQ(c.hsum(), 5.f);
    EXPECT_FLOAT_EQ(d.hsum(), 6.f);
    EXPECT_FLOAT_EQ(e.hsum(), 4.f);
    EXPECT_FLOAT_EQ(f.hsum(), 2.f);
}