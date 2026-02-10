#include <gtest/gtest.h>

#include <vectra/vectra.hpp>

TEST(VectratypeSSE41Float, Arithmetic)
{
    using vct = vectra::Vectratype<float, vectra::SIMDLevel::SSE41>;

    vct a(2.f, 3.f, 4.f, 5.f);
    vct b(3.f, 4.f, 5.f, 6.f);

    vct c = a + b;
    vct d = a * b;
    vct e = b - a;
    vct f = a / b;

    EXPECT_FLOAT_EQ(c.hsum(), 32.f);
    EXPECT_FLOAT_EQ(d.hsum(), 68.f);
    EXPECT_FLOAT_EQ(e.hsum(), 4.f);
    EXPECT_FLOAT_EQ(f.hsum(), 3.05f);
}