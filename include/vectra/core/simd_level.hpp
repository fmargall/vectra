#pragma once


#include <cstdint>


namespace vectra
{
	enum class SIMDLevel : std::uint8_t {
		None   = 0,
		SSE    = 1,
		SSE2   = 2,
		SSE3   = 3,
		SSSE3  = 4,
		SSE41  = 5,
		SSE42  = 6,
		AVX    = 7,
		AVX2   = 8,
		AVX512 = 9
	};
}