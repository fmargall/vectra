#pragma once


#include <vectra/backend/compute_backend.hpp>
#include <vectra/core/attributes.hpp>
#include <vectra/core/simd_level.hpp>


namespace vectra
{

template <typename T, SIMDLevel level>
struct Vectratype
{
	using backend = ComputeBackend<T, level>;
	using type    = typename backend::type;

	type value;

	FORCE_INLINE Vectratype() = default;
	
	// Native constructor only if type != T
	template<typename U = type,
		     typename = std::enable_if_t<!std::is_same_v<U, T>>>
	FORCE_INLINE explicit Vectratype(type x)   noexcept : value(x) {}

    /*
     * @brief SIMD lane-wise constructor.
     *
     * Constructs a Vectratype from exactly backend::width() scalar values.
     *
     * This constructor is only enabled when:
     *  - The number of arguments equals backend::width()
     *  - backend::width() > 1 (i.e. actual SIMD mode)
     *
     * Example (SSE, width = 4):
     *     Vectratype<float, SIMDLevel::SSE41> v(1.f, 2.f, 3.f, 4.f);
     *
     * Example (AVX2, width = 8):
     *     Vectratype<float, SIMDLevel::AVX2> v(a0, a1, ..., a7);
     *
     * Internally:
     *  - The arguments are stored in a stack-allocated temporary array
     *    aligned to backend::alignment().
     *  - The backend::load() function is then used to construct the SIMD register.
     *
     * Performance notes:
     *  - Fully inlined.
     *  - No heap allocation.
     *  - The temporary array is optimized away by the compiler.
     *  - Generates efficient _mm_set* or load instructions.
     *
     * Safety:
     *  - Enforced at compile-time: wrong number of arguments => compilation error.
     *  - Disabled for scalar (ie., SIMDLevel::None) to avoid constructor ambiguity
     */
	template<typename... Args,
			 typename = std::enable_if_t<(sizeof...(Args) == backend::width()) && (backend::width() > 1)>>
		FORCE_INLINE explicit Vectratype(Args... args) noexcept
	{
		alignas(backend::alignment()) T tmp[backend::width()] = {
			static_cast<T>(args)...
		};

		value = backend::loadu(tmp);
	}
	
	// Scalar constructor
	FORCE_INLINE explicit Vectratype(T scalar) noexcept : value(backend::set(scalar)) {}

	FORCE_INLINE friend Vectratype operator+(Vectratype a, Vectratype b) noexcept { return Vectratype(backend::add(a.value, b.value)); }
	FORCE_INLINE friend Vectratype operator-(Vectratype a, Vectratype b) noexcept { return Vectratype(backend::sub(a.value, b.value)); }
	FORCE_INLINE friend Vectratype operator*(Vectratype a, Vectratype b) noexcept { return Vectratype(backend::mul(a.value, b.value)); }
	FORCE_INLINE friend Vectratype operator/(Vectratype a, Vectratype b) noexcept { return Vectratype(backend::div(a.value, b.value)); }

    FORCE_INLINE static Vectratype sin (Vectratype x) noexcept { return Vectratype(backend::sin (x.value)); }
    FORCE_INLINE static Vectratype cos (Vectratype x) noexcept { return Vectratype(backend::cos (x.value)); }
    FORCE_INLINE static Vectratype acos(Vectratype x) noexcept { return Vectratype(backend::acos(x.value)); }
    FORCE_INLINE static Vectratype sqrt(Vectratype x) noexcept { return Vectratype(backend::sqrt(x.value)); }
    
    FORCE_INLINE static Vectratype min(Vectratype a, Vectratype b) noexcept { return Vectratype(backend::min(a.value, b.value)); }

    FORCE_INLINE static constexpr Vectratype two_pi() noexcept { return Vectratype(backend::two_pi()); }

    // Horizontal sum function. Useless for scalar
    // data, added here for maximum compatibility.
	FORCE_INLINE T hsum() const noexcept { return backend::hsum(value); }

    // Returns the SIMD register width, in terms of
    // the number of elements processed in parallel
    FORCE_INLINE static constexpr size_t width() noexcept { return backend::width(); }

    // Returns the memory alignment for the register
    FORCE_INLINE static constexpr size_t alignment() noexcept { return backend::alignment(); }

    // Loads value from pointer. Useless for scalar
    // data, added here for complete compatibility.
    FORCE_INLINE static Vectratype loadu(const T* ptr) noexcept { return Vectratype(backend::loadu(ptr)); }
    FORCE_INLINE static Vectratype loada(const T* ptr) noexcept { return Vectratype(backend::loadu(ptr)); }
};

}