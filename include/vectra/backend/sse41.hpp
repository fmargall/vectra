#pragma once

#include <immintrin.h>

#include <vectra/core/simd_level.hpp>
#include <vectra/core/attributes.hpp>
#include <vectra/core/constants.hpp>


namespace vectra
{

template <>
struct ComputeBackend<float, SIMDLevel::SSE41> {
	using type = __m128;
	FORCE_INLINE static type sin (type x)		  noexcept { return _mm_sin_ps(x); }
	FORCE_INLINE static type cos (type x)		  noexcept { return _mm_cos_ps(x); }
	// Since our approximation of arccos is not defined only over
	// [-1 ; 1], we can then avoid the cost of clamping argument.
	#ifndef HAS_MM_ACOS_PS
	FORCE_INLINE static type acos(type x)		  noexcept { return _mm_acos_ps(x); }
	#else
	FORCE_INLINE static type acos(type x)		  noexcept { return _mm_acos_ps(_mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-1.f)), _mm_set1_ps(1.f))); }
	#endif
	FORCE_INLINE static type sqrt(type x)		  noexcept { return _mm_sqrt_ps(x); }
	FORCE_INLINE static type cbrt(type x)		  noexcept { return _mm_cbrt_ps(x); }
	FORCE_INLINE static type exp (type x)		  noexcept { return _mm_exp_ps(x); }
	FORCE_INLINE static type add (type a, type b) noexcept { return _mm_add_ps(a, b); }
	FORCE_INLINE static type sub (type a, type b) noexcept { return _mm_sub_ps(a, b); }
	FORCE_INLINE static type mul (type a, type b) noexcept { return _mm_mul_ps(a, b); }
	FORCE_INLINE static type div (type a, type b) noexcept { return _mm_div_ps(a, b); }
	FORCE_INLINE static type min (type a, type b) noexcept { return _mm_min_ps(a, b); }

	FORCE_INLINE static type one()				  noexcept { return _mm_set1_ps(1.f); }
	FORCE_INLINE static type zero()				  noexcept { return _mm_setzero_ps(); }
	FORCE_INLINE static type two_pi()			  noexcept { return _mm_set1_ps(TWO_PI_F); }

	// Conversion function from scalar value to SIMD
	FORCE_INLINE static type set(float x) noexcept { return _mm_set1_ps(x); }

	// Returns the SIMD register width, in terms of
	// the number of elements processed in parallel
	// NB: this implementation is faster than using
	//     the _mm_hadd_ps
	// Source: stackoverflow.com/a/35270026/8885740
	//         Peter Cordes - 2016
	FORCE_INLINE static float hsum(type x) noexcept {
		__m128 shuf = _mm_movehdup_ps(x);
		__m128 sums = _mm_add_ps(x, shuf);
		shuf  = _mm_movehl_ps(shuf, sums);
		sums  = _mm_add_ss(sums, shuf);
		return  _mm_cvtss_f32(sums);
	}

	// Returns the SIMD register width, in terms of
	// the number of elements processed in parallel
	FORCE_INLINE static constexpr size_t width() noexcept { return 4; }

	// Returns the memory alignment for the register 
	FORCE_INLINE static constexpr size_t alignment() noexcept { return alignof(type); }

	// Loads value from pointer to associated data type
	FORCE_INLINE static type loadu(const float* FORCE_RESTRICT ptr) noexcept { return _mm_loadu_ps(ptr); } // Unaligned
	FORCE_INLINE static type loada(const float* FORCE_RESTRICT ptr) noexcept { return _mm_load_ps (ptr); } // Aligned

	// Unloads SIMD value to scalar buffers
	FORCE_INLINE static void unloadu(float* FORCE_RESTRICT ptr, type x) { _mm_storeu_ps(ptr, x); } // Unaligned
	FORCE_INLINE static void unloada(float* FORCE_RESTRICT ptr, type x) { _mm_store_ps (ptr, x); } // Aligned

};

}