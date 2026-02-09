#pragma once

#include <vectra/core/simd_level.hpp>

namespace vectra
{

template <typename FloatingPrecision, SIMDLevel simdLevel>
struct ComputeBackend;

}

#include <vectra/backend/none.hpp>
#include <vectra/backend/sse41.hpp>