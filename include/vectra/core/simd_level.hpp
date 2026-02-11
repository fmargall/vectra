#pragma once


#include <cstdint>
#include <ostream>


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

constexpr const char* toString(SIMDLevel level)
{
    switch (level)
    {
        case SIMDLevel::None  : return "None";
        case SIMDLevel::SSE   : return "SSE";
        case SIMDLevel::SSE2  : return "SSE2";
        case SIMDLevel::SSE3  : return "SSE3";
        case SIMDLevel::SSSE3 : return "SSSE3";
        case SIMDLevel::SSE41 : return "SSE4.1";
        case SIMDLevel::SSE42 : return "SSE4.2";
        case SIMDLevel::AVX   : return "AVX";
        case SIMDLevel::AVX2  : return "AVX2";
        case SIMDLevel::AVX512: return "AVX512";
    }
    return "Unknown";
}

inline std::ostream& operator<<(std::ostream& os, SIMDLevel level)
{
    return os << toString(level);
}

}