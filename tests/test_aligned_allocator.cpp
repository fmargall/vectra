#include <cstdint>
#include <vector>

#include <immintrin.h>

#include <gtest/gtest.h>

#include <vectra/vectra.hpp>


TEST(AlignedAllocator, FloatVectorLoadedAsSSE41)
{
    using aligned_float_vector =
        std::vector<float, vectra::aligned_allocator<float, 16>>;

    aligned_float_vector data(8);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>(i);

    const auto addr = reinterpret_cast<std::uintptr_t>(data.data());
    EXPECT_EQ(addr % 16, 0u);

    const __m128 v = _mm_load_ps(data.data());
    const __m128 doubled = _mm_add_ps(v, v);

    alignas(16) float out[4];
    _mm_store_ps(out, doubled);

    EXPECT_FLOAT_EQ(out[0], 0.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
    EXPECT_FLOAT_EQ(out[2], 4.f);
    EXPECT_FLOAT_EQ(out[3], 6.f);
}

TEST(AlignedAllocator, M128VectorStorage)
{
    using m128_vector =
        std::vector<
        __m128,
        vectra::aligned_allocator<__m128, 16>
        >;

    m128_vector data(2);

    __m128 v0 = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
    __m128 v1 = _mm_set_ps(7.f, 6.f, 5.f, 4.f);

    data[0] = v0;
    data[1] = v1;

    auto addr = reinterpret_cast<std::uintptr_t>(data.data());
    EXPECT_EQ(addr % 16, 0u);

    alignas(16) float result[4];

    _mm_store_ps(result, data[0]);
    EXPECT_FLOAT_EQ(result[0], 0.f);
    EXPECT_FLOAT_EQ(result[1], 1.f);
    EXPECT_FLOAT_EQ(result[2], 2.f);
    EXPECT_FLOAT_EQ(result[3], 3.f);

    _mm_store_ps(result, data[1]);
    EXPECT_FLOAT_EQ(result[0], 4.f);
    EXPECT_FLOAT_EQ(result[1], 5.f);
    EXPECT_FLOAT_EQ(result[2], 6.f);
    EXPECT_FLOAT_EQ(result[3], 7.f);
}