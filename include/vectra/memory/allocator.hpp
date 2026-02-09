#pragma once

namespace vectra
{

/*
 * @brief STL-compatible aligned allocator.
 *
 * This allocator guarantees that allocated memory blocks are aligned
 * to the specified byte boundary (Alignment).
 *
 * It relies on C++17 aligned operator new/delete and is therefore:
 *  - Portable across MSVC / GCC / Clang
 *  - Fully compatible with std::vector
 *  - Zero-overhead in practice (allocation is rare)
 *
 * This is required for SIMD types such as:
 *  - __m128  (16-byte alignment)
 *  - __m256  (32-byte alignment)
 *  - __m512  (64-byte alignment)
 *
 * IMPORTANT:
 * Alignment must be:
 *  - A power of two
 *  - >= alignof(T)
 *
 * This allocator is stateless and all instances are interchangeable.
 */
template <typename T, std::size_t Alignment>
class aligned_allocator
{
    // Alignment must be a power of two
    static_assert((Alignment& (Alignment - 1)) == 0,
        "Alignment must be a power of two.");

    // Ensure we do not under-align the type
    static_assert(Alignment >= alignof(T),
        "Alignment must be >= alignof(T).");

public:
	// Required typdefs for STL allocator compatibility
	using value_type = T;

    /*
     * These two traits inform the STL that:
     *
     * - All allocator instances are interchangeable
     * - Move assignment can propagate allocator safely
     *
     * This allows std::vector to:
     *  - Steal internal buffers on move,
     *  - Avoid unnecessary reallocations
     *
     * Since this allocator is stateless, this is correct.
     */
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    // Default constructor (stateless allocator)
    aligned_allocator() noexcept = default;

    /**
     * Templated conversion constructor.
     *
     * Required so that STL can convert between
     * aligned_allocator<U, Alignment>
     * and
     * aligned_allocator<T, Alignment>.
     *
     * This is needed internally by std::vector
     * when rebinding allocators to different types.
     *
     * Even though our allocator is stateless,
     * this constructor ensures maximum STL compatibility.
     */
	template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    /*
     * @brief Allocate memory for n elements.
     *
     * @param n Number of elements to allocate.
     *
     * @return Pointer to aligned memory block.
     *
     * Uses C++17 aligned operator new:
     *   ::operator new(size, std::align_val_t{Alignment})
     *
     * This ensures:
     *   - Proper alignment
     *   - Portable behavior
     *   - Standard-compliant allocation
     *
     * Throws std::bad_alloc on failure.
     */
    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n == 0)
            return nullptr;

        const std::size_t bytes = n * sizeof(T);

        return static_cast<T*>(
            ::operator new(bytes, std::align_val_t{ Alignment })
        );
    }

    /*
     * @brief Deallocate previously allocated memory
     */
    void deallocate(T* ptr, std::size_t) noexcept
    {
        ::operator delete(ptr, std::align_val_t{ Alignment });
    }

    /*
     * Required by older STL implementations
     */
    template<typename U>
    struct rebind
    {
        using other = aligned_allocator<U, Alignment>;
    };

};


/*
 * Allocators are equal if their alignment is equal.
 * Since they are stateless, equality depends only on Alignment.
 */
template<typename T1, std::size_t A1,
         typename T2, std::size_t A2>
constexpr bool operator==(const aligned_allocator<T1, A1>&,
                          const aligned_allocator<T2, A2>&) noexcept
{
    return A1 == A2;
}

template<typename T1, std::size_t A1,
         typename T2, std::size_t A2>
constexpr bool operator!=(const aligned_allocator<T1, A1>& a,
                          const aligned_allocator<T2, A2>& b) noexcept
{
    return !(a == b);
}


}