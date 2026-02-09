#pragma once


#include "vectra/core/simd_level.hpp"
#include "vectra/detail/cpuid.hpp"


namespace vectra
{
	inline SIMDLevel highestRuntimeSIMDLevel() {

		// Vectra currently only supports x86-64 architecture 
		// For other architectures, returns SIMDLevel::None.
		#if defined(__x86_64__) || defined(_M_X64)
			// CPUID, when called with EAX=0, returns the maximum function ID
			// that can then be called. To continue we will need access to 1,
			// and 7. If not, SIMD instructions will be disabled.
			std::uint32_t info[4];
			detail::cpuid(info, 0);
			std::uint32_t maxID = info[0];

			if (maxID >= 1) {
				// All the following bit instructions are manufacturer specific,
				// but usually there are standardized for sake of compatibility.
				// Source can be found on en.wikipedia.org/wiki/CPUID

				// Runs the CPUID instruction, reads and stores the CPU registers
				// in the info array. Each bit can be used to read compatibility.
				// Until AVX, all features are stored in the EAX=1 register.
				detail::cpuid(info, 1);

				//   Feature |  Register |   Bit
				//   SSE     |  EDX      |   Bit 25
				bool sse     = (info[3]  & (1 << 25)) != 0;
				//   SSE2    |  EDX      |   Bit 26
				bool sse2    = (info[3]  & (1 << 26)) != 0;
				//   SSE3    |  ECX      |   Bit 0
				bool sse3    = (info[2]  & (1 << 0))  != 0;
				//   SSSE3   |  ECX      |   Bit 9
				bool ssse3   = (info[2]  & (1 << 9))  != 0;
				//   SSE4.1  |  ECX      |   Bit 19
				bool sse41   = (info[2]  & (1 << 19)) != 0;
				//   SSE4.2  |  ECX      |   Bit 20
				bool sse42   = (info[2]  & (1 << 20)) != 0;
				//   AVX     |  ECX      |   Bit 28
				bool avx     = (info[2]  & (1 << 28)) != 0;

				// The OS must support XSAVE to use AVX and later instructions.
				// This should also be checked before using AVX, AVX2 or AVX512

				//   XSAVE   |  ECX     |   Bit 27
				bool osxsave = (info[2] & (1 << 27)) != 0;

				// AVX2 and AVX512 are stored in the EAX=7, ECX=0 register.
				detail::cpuid(info, 7, 0);

				// Most of the AVX-512 functions needed belong to the AVX512F
				// (Foundation) instruction set, that is associated to bit 16
				// But another instruction that we will use (_mm512_and_ps/d)
				// will need the AVX512DQ (Doubleword & Quadword Instruction)
				// set, associated to the bit 17.

				//   Feature  |  Register |   Bit
				//   AVX2     |  EBX      |   Bit 5
				bool avx2     = (info[1]  & (1 << 5))  != 0;
				//   AVX512F  |  EBX      |   Bit 16
				bool avx512f  = (info[1]  & (1 << 16)) != 0;
				//   AVX512DQ |  EBX      |   Bit 17
				bool avx512dq = (info[1]  & (1 << 17)) != 0;

				// In order to use AVX and later, one must check XCR0, i.e. the
				// Extended Control Register 0 to ensure that the XMM, YMM, and
				// ZMM registers are enabled.
				// Source can be found: en.wikipedia.org/wiki/Control_register

				// AVX2 and AVX512 use the XMM, YMM and ZMM registers. We could
				// check their accessibility by reading the XCR0 register using
				// the bits 1 (XMM), 2 (YMM) and 5, 6, 7 (ZMM). Let's apply the
				// mask 1110 0110 (0xE6 in hexadecimal) and check its value.
				bool avx2Registers = (detail::xgetbv(0) & 0xE6) == 0xE6;
				// AVX, AVX2 and AVX512 all use the XMM and YMM registers. They
				// have their accessibility encoded in the XCR0 register at the
				// bits 1 for XMM and 2 for YMM. Let's apply the mask 0000 0110
				// (0x6 in hexadecimal) to the XCR0 and check its value.
				bool avxRegister = (detail::xgetbv(0) & 0x6) == 0x6;

				if (avx512f && avx512dq && osxsave && avx2Registers) return SIMDLevel::AVX512;
				if (avx2                && osxsave && avx2Registers) return SIMDLevel::AVX2;
				if (avx                 && osxsave && avxRegister)   return SIMDLevel::AVX;
				if (sse42)							 	             return SIMDLevel::SSE42;
				if (sse41)						 					 return SIMDLevel::SSE41;
				if (ssse3)											 return SIMDLevel::SSSE3;
				if (sse3)											 return SIMDLevel::SSE3;
				if (sse2)											 return SIMDLevel::SSE2;
				if (sse)											 return SIMDLevel::SSE;
				else												 return SIMDLevel::None;

			}
			else
				return SIMDLevel::None;
		
		// ARM and other architectures are not supported yet
		#else
			return SIMDLevel::None;
		#endif
	}
}