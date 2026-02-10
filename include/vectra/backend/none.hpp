#pragma once


#include <cmath>


#include <vectra/core/simd_level.hpp>
#include <vectra/core/attributes.hpp>
#include <vectra/core/constants.hpp>


namespace vectra {

template <>
struct ComputeBackend<float, SIMDLevel::None> {
	using type = float;
	FORCE_INLINE static type sin (type x)		  noexcept { return std::sin(x); }
	FORCE_INLINE static type cos (type x)		  noexcept { return std::cos(x); }
	FORCE_INLINE static type acos(type x)		  noexcept { return std::acos(x); }
	FORCE_INLINE static type sqrt(type x)		  noexcept { return std::sqrt(x); }
	FORCE_INLINE static type cbrt(type x)		  noexcept { return std::cbrt(x); }
	FORCE_INLINE static type exp (type x)		  noexcept { return std::exp(x); }
	FORCE_INLINE static type add (type a, type b) noexcept { return a + b; }
	FORCE_INLINE static type sub (type a, type b) noexcept { return a - b; }
	FORCE_INLINE static type mul (type a, type b) noexcept { return a * b; }
	FORCE_INLINE static type div (type a, type b) noexcept { return a / b; }
	FORCE_INLINE static type min (type a, type b) noexcept { return a < b ? a : b; }

	FORCE_INLINE static constexpr type one()	  noexcept { return 1.f; }
	FORCE_INLINE static constexpr type zero()	  noexcept { return 0.f; }
	FORCE_INLINE static constexpr type two_pi()	  noexcept { return TWO_PI_F; }

	// Conversion function. Useless for scalar data
	// but is added here for maximum compatibility.
	FORCE_INLINE static type set(type x) noexcept { return x; }

	// Horizontal sum function. Useless for scalar
	// data, added here for maximum compatibility.
	FORCE_INLINE static type hsum(type x) noexcept { return x; }

	// Returns the SIMD register width, in terms of
	// the number of elements processed in parallel
	FORCE_INLINE static constexpr size_t width() noexcept { return 1; }

	// Returns the memory alignment for the register 
	FORCE_INLINE static constexpr size_t alignment() noexcept { return alignof(type); }

	// Loads value from pointer. Useless for scalar
	// data, added here for complete compatibility.
	FORCE_INLINE static type loadu(const float* ptr) noexcept { return *ptr; }
	FORCE_INLINE static type loada(const float* ptr) noexcept { return *ptr; }

	// Unloads SIMD value to scalar buffers. Useless
	// for scalar data, added here for compatibility
	FORCE_INLINE static void unloadu(float* ptr, type x) noexcept { *ptr = x; }
	FORCE_INLINE static void unloada(float* ptr, type x) noexcept { *ptr = x; }

};

template <>
struct ComputeBackend<double, SIMDLevel::None> {
	using type = double;
	FORCE_INLINE static type sin (type x)		  noexcept { return std::sin (x); }
	FORCE_INLINE static type cos (type x)		  noexcept { return std::cos (x); }
	FORCE_INLINE static type acos(type x)		  noexcept { return std::acos(x); }
	FORCE_INLINE static type sqrt(type x)		  noexcept { return std::sqrt(x); }
	FORCE_INLINE static type cbrt(type x)		  noexcept { return std::cbrt(x); }
	FORCE_INLINE static type exp (type x)		  noexcept { return std::exp (x); }
	FORCE_INLINE static type add (type a, type b) noexcept { return a + b; }
	FORCE_INLINE static type sub (type a, type b) noexcept { return a - b; }
	FORCE_INLINE static type mul (type a, type b) noexcept { return a * b; }
	FORCE_INLINE static type div (type a, type b) noexcept { return a / b; }
	FORCE_INLINE static type min (type a, type b) noexcept { return a < b ? a : b; }

	FORCE_INLINE static constexpr type one()	  noexcept { return 1.; }
	FORCE_INLINE static constexpr type zero()	  noexcept { return 0.; }
	FORCE_INLINE static constexpr type two_pi()	  noexcept { return TWO_PI_D; }

	// Conversion function. Useless for scalar data,
	// added here for maximum compatibility.
	FORCE_INLINE static double set(double x) noexcept { return x; }

	// Horizontal sum function. Useless for scalar
	// data, added here for maximum compatibility.
	FORCE_INLINE static double hsum(type x) noexcept { return x; }

	// Returns the SIMD register width, in terms of
	// the number of elements processed in parallel
	FORCE_INLINE static constexpr size_t width() noexcept { return 1; }

	// Returns the memory alignment for the register
	FORCE_INLINE static constexpr size_t alignment() noexcept { return alignof(type); }

	// Loads value from pointer. Useless for scalar
	// data, added here for complete compatibility.
	FORCE_INLINE static type loadu(const double* ptr) noexcept { return *ptr; }
	FORCE_INLINE static type loada(const double* ptr) noexcept { return *ptr; }

	// Unloads SIMD value to scalar buffers. Useless
	// for scalar data, added here for compatibility
	FORCE_INLINE static void unloadu(double* ptr, type x) noexcept { *ptr = x; }
	FORCE_INLINE static void unloada(double* ptr, type x) noexcept { *ptr = x; }
};

}