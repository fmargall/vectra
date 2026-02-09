#pragma once

// Defines macro for forcing inlining based on the compiler
// It is absolutely required to achieve better performance.
#if defined(_MSC_VER)
	#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
	#define FORCE_INLINE inline __attribute__((always_inline))
#else
	#define FORCE_INLINE inline
#endif

// Defines macro for forcing the use of the restrict qualifier,
// based on the compiler. Can help the compiler optimize memory
#if defined(_MSC_VER)
	#define FORCE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define FORCE_RESTRICT __restrict__
#else
	#define FORCE_RESTRICT
#endif