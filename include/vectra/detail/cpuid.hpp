#pragma once


#include <cstdint>


// Vectra currently only supports x86-64 architecture 
#if defined(__x86_64__) || defined(_M_X64)

	#if   defined(_MSC_VER)
		#include <intrin.h>
	#elif defined(__GNUC__) || defined(__clang__)
		#include <cpuid.h>
	#endif


	namespace vectra::detail
	{
		/*
		 * @brief Executes the CPUID instruction.
		 *
		 * Queries CPU feature information using the CPUID instruction
		 * The result registers (EAX, EBX, ECX, EDX) are stored in the
		 * provided array in that order.
		 *
		 * This function abstracts compiler-specific intrinsics:
		 *  - MSVC     : __cpuidex
		 *  - GCC/Clang: __cpuid_count
		 *
		 * @param info          Output array of 4 elements receiving
		 *                      {EAX, EBX, ECX, EDX}.
		 * @param functionID    CPUID leaf (EAX input).
		 * @param subfunctionID CPUID sub-leaf (ECX input), default = 0
		 *
		 * @note If the compiler is unsupported, all output registers
		 *       are set to zero.
		 *
		 * @warning This function does not validate the availability
		 *          of the requested leaf. Callers should ensure the
		 *          leaf is supported (e.g. via CPUID(0)).
		 */
		inline void cpuid(std::uint32_t info[4], std::uint32_t functionID, std::uint32_t subfunctionID = 0) {
			#if defined(_MSC_VER)
				int regs[4];
				__cpuidex(regs,
					static_cast<int>(functionID),
					static_cast<int>(subfunctionID));

				info[0] = static_cast<std::uint32_t>(regs[0]);
				info[1] = static_cast<std::uint32_t>(regs[1]);
				info[2] = static_cast<std::uint32_t>(regs[2]);
				info[3] = static_cast<std::uint32_t>(regs[3]);

			#elif defined(__GNUC__) || defined(__clang__)
				__cpuid_count(functionID, subfunctionID, info[0], info[1], info[2], info[3]);

			#else
				// Unsupported compiler. SIMD instructions will be disabled.
				info[0] = 0;
				info[1] = 0;
				info[2] = 0;
				info[3] = 0;

			#endif
		}

		/*
		 * @brief Reads an extended control register using XGETBV.
		 *
		 * Returns the value of the specified extended control register,
		 * (typically XCR0) which encodes the OS-enabled processor state
		 *
		 * This is required to determine whether SIMD register states:
		 * (XMM, YMM, ZMM, opmask) are enabled by the operating system
		 *
		 * Compiler-specific implementations:
		 *  - MSVC     : _xgetbv intrinsic
		 *  - GCC/Clang: inline assembly
		 *
		 * @param index Index of the extended control register (usually 0)
		 * @return 64-bit value of the requested XCR register
		 *
		 * @warning Must only be called if CPUID reports OSXSAVE support
		 *          Calling this, without OSXSAVE enabled may trigger an
		 *          illegal instruction exception.
		 */
		inline uint64_t xgetbv(uint32_t index) {
			#if   defined(_MSC_VER)
				return _xgetbv(index);

			#elif defined(__GNUC__) || defined(__clang__)
				uint32_t eax, edx;
				__asm__ volatile (
					"xgetbv"
					: "=a"(eax), "=d"(edx)
					: "c"(index)
					);
				return ((uint64_t)edx << 32) | eax;

			#else
				// Unsupported compiler. SIMD instructions will be disabled.
				return 0;

			#endif
		}
	}

#endif // Functions defined for x86-64 only