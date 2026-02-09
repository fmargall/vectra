#pragma once


#include <vectra/backend/compute_backend.hpp>
#include <vectra/core/attributes.hpp>
#include <vectra/core/simd_level.hpp>


namespace vectra
{

template <typename T, SIMDLevel level>
struct Pack
{
	using backend = ComputeBackend<T, level>;
	using type    = typename backend::type;

	type value;

	FORCE_INLINE Pack() = default;
	
	// Native constructor only if type != T
	template<typename U = type,
		     typename = std::enable_if_t<!std::is_same_v<U, T>>>
	FORCE_INLINE explicit Pack(type x)   noexcept : value(x) {}
	
	// Scalar constructor
	FORCE_INLINE explicit Pack(T scalar) noexcept : value(backend::set(scalar)) {}

	FORCE_INLINE friend Pack operator+(Pack a, Pack b) noexcept { return Pack(backend::add(a.value, b.value)); }
	FORCE_INLINE friend Pack operator-(Pack a, Pack b) noexcept { return Pack(backend::sub(a.value, b.value)); }
	FORCE_INLINE friend Pack operator*(Pack a, Pack b) noexcept { return Pack(backend::mul(a.value, b.value)); }
	FORCE_INLINE friend Pack operator/(Pack a, Pack b) noexcept { return Pack(backend::div(a.value, b.value)); }

	FORCE_INLINE T hsum() const noexcept { return backend::hsum(value); }
};

}