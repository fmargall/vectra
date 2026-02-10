#pragma once

// Backend dispatch header. This is the only 
// header that needs to be included by users
#include <vectra/backend/compute_backend.hpp>

// Aligned memory allocator header.
#include <vectra/memory/allocator.hpp>

// Vectratype header. This is the main 
// type of the library, easier to use.
#include <vectra/types/vectratype.hpp>

// Runtime checks header. There is no need to
// include cpuid.hpp or any other header that
// is inside the detail namespace.
#include <vectra/dispatch/runtime_checks.hpp>