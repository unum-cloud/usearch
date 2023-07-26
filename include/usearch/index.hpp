/**
 *  @file index.hpp
 *  @author Ash Vardanian
 *  @brief Single-header Vector Search.
 *  @date 2023-04-26
 *
 *  @copyright Copyright (c) 2023
 */
#ifndef UNUM_USEARCH_HPP
#define UNUM_USEARCH_HPP

#define USEARCH_VERSION_MAJOR 0
#define USEARCH_VERSION_MINOR 0
#define USEARCH_VERSION_PATCH 0

// Inferring C++ version
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define USEARCH_DEFINED_CPP17
#endif

// Inferring target OS
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define USEARCH_DEFINED_WINDOWS
#elif defined(__APPLE__) && defined(__MACH__)
#define USEARCH_DEFINED_APPLE
#elif defined(__linux__)
#define USEARCH_DEFINED_LINUX
#endif

// Inferring the compiler
#if defined(__clang__)
#define USEARCH_DEFINED_CLANG
#elif defined(__GNUC__)
#define USEARCH_DEFINED_GCC
#endif

// Inferring hardware architecture: x86 vs Arm
#if defined(__x86_64__)
#define USEARCH_DEFINED_X86
#elif defined(__aarch64__)
#define USEARCH_DEFINED_ARM
#endif

// Inferring hardware bitness: 32 vs 64
#if __WORDSIZE == 64
#define USEARCH_64BIT_ENV
#else
#define USEARCH_32BIT_ENV
#endif

#if !defined(USEARCH_USE_OPENMP)
#define USEARCH_USE_OPENMP 0
#endif

// OS-specific includes
#if defined(USEARCH_DEFINED_WINDOWS)
#define _USE_MATH_DEFINES
#define NOMINMAX
#include <Windows.h>
#include <sys/stat.h> // `fstat` for file size
#undef NOMINMAX
#undef _USE_MATH_DEFINES
#else
#include <fcntl.h>    // `fallocate`
#include <stdlib.h>   // `posix_memalign`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `fstat` for file size
#include <unistd.h>   // `open`, `close`
#endif

// STL includes
#include <algorithm> // `std::sort_heap`
#include <atomic>    // `std::atomic`
#include <bitset>    // `std::bitset`
#include <climits>   // `CHAR_BIT`
#include <cmath>     // `std::sqrt`
#include <cstring>   // `std::memset`
#include <iterator>  // `std::reverse_iterator`
#include <mutex>     // `std::unique_lock` - replacement candidate
#include <random>    // `std::default_random_engine` - replacement candidate
#include <stdexcept> // `std::runtime_exception`
#include <thread>    // `std::thread`
#include <utility>   // `std::pair`

// Prefetching
#if defined(USEARCH_DEFINED_GCC)
// https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
// Zero means we are only going to read from that memory.
// Three means high temporal locality and suggests to keep
// the data in all layers of cache.
#define prefetch_m(ptr) __builtin_prefetch((void*)(ptr), 0, 3)
#elif defined(USEARCH_DEFINED_X86)
#define prefetch_m(ptr) _mm_prefetch((void*)(ptr), _MM_HINT_T0)
#else
#define prefetch_m(ptr)
#endif

// Alignment
#if defined(USEARCH_DEFINED_WINDOWS)
#define usearch_pack_m
#define usearch_align_m __declspec(align(64))
#else
#define usearch_pack_m __attribute__((packed))
#define usearch_align_m __attribute__((aligned(64)))
#endif

// Debugging
#if defined(NDEBUG)
#define usearch_assert_m(must_be_true, message)
#define usearch_noexcept_m noexcept
#else
#define usearch_assert_m(must_be_true, message)                                                                        \
    if (!(must_be_true)) {                                                                                             \
        throw std::runtime_error(message);                                                                             \
    }
#define usearch_noexcept_m
#endif

namespace unum {
namespace usearch {

using byte_t = char;

template <std::size_t multiple_ak> std::size_t divide_round_up(std::size_t num) noexcept {
    return (num + multiple_ak - 1) / multiple_ak;
}

inline std::size_t divide_round_up(std::size_t num, std::size_t denominator) noexcept {
    return (num + denominator - 1) / denominator;
}

inline std::size_t ceil2(std::size_t v) noexcept {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#ifdef USEARCH_64BIT_ENV
    v |= v >> 32;
#endif
    v++;
    return v;
}

/// @brief  Simply dereferencing misaligned pointers can be dangerous.
template <typename at> void misaligned_store(void* ptr, at v) noexcept {
    static_assert(!std::is_reference<at>::value, "Can't store a reference");
    std::memcpy(ptr, &v, sizeof(at));
}

/// @brief  Simply dereferencing misaligned pointers can be dangerous.
template <typename at> at misaligned_load(void* ptr) noexcept {
    static_assert(!std::is_reference<at>::value, "Can't load a reference");
    at v;
    std::memcpy(&v, ptr, sizeof(at));
    return v;
}

/// @brief  The `std::exchange` alternative for C++11.
template <typename at, typename other_at = at> at exchange(at& obj, other_at&& new_value) {
    at old_value = std::move(obj);
    obj = std::forward<other_at>(new_value);
    return old_value;
}

/**
 *  @brief  A reference to a misaligned memory location with a specific type.
 *          It is needed to avoid Undefiend Behavior when dereferencing addresses
 *          indivisible by `sizeof(at)`.
 */
template <typename at> class misaligned_ref_gt {
    byte_t* ptr_;

  public:
    misaligned_ref_gt(byte_t* ptr) noexcept : ptr_(ptr) {}
    operator at() const noexcept { return misaligned_load<at>(ptr_); }
    misaligned_ref_gt& operator=(at const& v) noexcept {
        misaligned_store<at>(ptr_, v);
        return *this;
    }

    void reset(byte_t* ptr) noexcept { ptr_ = ptr; }
};

/**
 *  @brief  A pointer to a misaligned memory location with a specific type.
 *          It is needed to avoid Undefiend Behavior when dereferencing addresses
 *          indivisible by `sizeof(at)`.
 */
template <typename at> class misaligned_ptr_gt {
    byte_t* ptr_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = at;
    using difference_type = std::ptrdiff_t;
    using pointer = misaligned_ptr_gt<at>;
    using reference = misaligned_ref_gt<at>;

    reference operator*() const noexcept { return {ptr_}; }
    reference operator[](std::size_t i) noexcept { return reference(ptr_ + i * sizeof(at)); }
    value_type operator[](std::size_t i) const noexcept { return misaligned_load<at>(ptr_ + i * sizeof(at)); }

    misaligned_ptr_gt(byte_t* ptr) noexcept : ptr_(ptr) {}
    misaligned_ptr_gt operator++(int) noexcept { return misaligned_ptr_gt(ptr_ + sizeof(at)); }
    misaligned_ptr_gt operator--(int) noexcept { return misaligned_ptr_gt(ptr_ - sizeof(at)); }
    misaligned_ptr_gt operator+(difference_type d) noexcept { return misaligned_ptr_gt(ptr_ + d * sizeof(at)); }
    misaligned_ptr_gt operator-(difference_type d) noexcept { return misaligned_ptr_gt(ptr_ - d * sizeof(at)); }

    misaligned_ptr_gt& operator++() noexcept {
        ptr_ += sizeof(at);
        return *this;
    }
    misaligned_ptr_gt& operator--() noexcept {
        ptr_ -= sizeof(at);
        return *this;
    }
    misaligned_ptr_gt& operator+=(difference_type d) noexcept {
        ptr_ += d * sizeof(at);
        return *this;
    }
    misaligned_ptr_gt& operator-=(difference_type d) noexcept {
        ptr_ -= d * sizeof(at);
        return *this;
    }
    bool operator==(misaligned_ptr_gt const& other) noexcept { return ptr_ == other.ptr_; }
    bool operator!=(misaligned_ptr_gt const& other) noexcept { return ptr_ != other.ptr_; }
};

/**
 *  @brief  Non-owning memory range view, similar to `std::span`, but for C++11.
 */
template <typename scalar_at> class span_gt {
    scalar_at* data_;
    std::size_t size_;

  public:
    span_gt() noexcept : data_(nullptr), size_(0u) {}
    span_gt(scalar_at* begin, scalar_at* end) noexcept : data_(begin), size_(end - begin) {}
    span_gt(scalar_at* begin, std::size_t size) noexcept : data_(begin), size_(size) {}
    scalar_at* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    scalar_at* begin() const noexcept { return data_; }
    scalar_at* end() const noexcept { return data_ + size_; }
    operator scalar_at*() const noexcept { return data(); }
};

/**
 *  @brief  A lightweight error class for handling error messages,
 *          which are expected to be allocated in static memory.
 */
class error_t {
    char const* message_{};

  public:
    error_t(char const* message = nullptr) noexcept : message_(message) {}
    error_t& operator=(char const* message) noexcept {
        message_ = message;
        return *this;
    }
    error_t(error_t&& other) noexcept : message_(exchange(other.message_, nullptr)) {}
    error_t& operator=(error_t&& other) noexcept {
        std::swap(message_, other.message_);
        return *this;
    }
    explicit operator bool() const noexcept { return message_ != nullptr; }
    char const* what() const noexcept { return message_; }

#if defined(__cpp_exceptions) && (1 == __cpp_exceptions)
    ~error_t() noexcept(false) {
        if (message_)
            if (!std::uncaught_exception())
                raise();
    }
    void raise() noexcept(false) {
        if (message_)
            throw std::runtime_error(exchange(message_, nullptr));
    }
#else
    ~error_t() noexcept { raise(); }
    void raise() noexcept {
        if (message_)
            std::terminate();
    }
#endif
};

/**
 *  @brief  Similar to `std::expected` in C++23, wraps a statement evaluation result,
 *          or an error. It's used to avoid raising exception, and gracefully propagate
 *          the error.
 *
 * @tparam result_at The type of the expected result.
 */
template <typename result_at> struct expected_gt {
    result_at result;
    error_t error;

    operator result_at&() & {
        error.raise();
        return result;
    }
    operator result_at&&() && {
        error.raise();
        return std::move(result);
    }
    result_at const& operator*() const noexcept { return result; }
    explicit operator bool() const noexcept { return !error; }
    expected_gt failed(error_t message) noexcept {
        error = std::move(message);
        return std::move(*this);
    }
};

/**
 *  @brief  Light-weight bitset implementation to track visited nodes during graph traversal.
 *          Extends basic functionality with @b atomic operations.
 */
template <typename allocator_at = std::allocator<byte_t>> class visits_bitset_gt {
    using allocator_t = allocator_at;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    using slot_t = unsigned long;

    static constexpr std::size_t bits_per_slot() { return sizeof(slot_t) * CHAR_BIT; }
    static constexpr slot_t bits_mask() { return sizeof(slot_t) * CHAR_BIT - 1; }

    slot_t* slots_{};
    /// @brief Number of slots.
    std::size_t count_{};

  public:
    visits_bitset_gt() noexcept {}
    ~visits_bitset_gt() noexcept { reset(); }
    void clear() noexcept { std::memset(slots_, 0, count_ * sizeof(slot_t)); }

    void reset() noexcept {
        if (slots_)
            allocator_t{}.deallocate((byte_t*)slots_, count_ * sizeof(slot_t));
        slots_ = nullptr;
        count_ = 0;
    }

    /**
     *  @brief  Resizes the bitset to accomodate a given number of @b bits.
     *  @return False, if memory allocation error was encounted. True, otherwise.
     */
    bool resize(std::size_t capacity) noexcept {

        std::size_t count = divide_round_up<bits_per_slot()>(capacity);
        if (count <= count_)
            return true;

        slot_t* slots = (slot_t*)allocator_t{}.allocate(count * sizeof(slot_t));
        if (!slots)
            return false;

        reset();
        count_ = count;
        slots_ = slots;
        clear();
        return true;
    }

    visits_bitset_gt(visits_bitset_gt&& other) noexcept {
        slots_ = exchange(other.slots_, nullptr);
        count_ = exchange(other.count_, 0);
    }

    visits_bitset_gt& operator=(visits_bitset_gt&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(count_, other.count_);
        return *this;
    }

    visits_bitset_gt(visits_bitset_gt const&) = delete;
    visits_bitset_gt& operator=(visits_bitset_gt const&) = delete;

    inline bool test(std::size_t i) const noexcept { return slots_[i / bits_per_slot()] & (1ul << (i & bits_mask())); }
    inline void set(std::size_t i) noexcept { slots_[i / bits_per_slot()] |= (1ul << (i & bits_mask())); }

#if defined(USEARCH_DEFINED_WINDOWS)

    inline bool atomic_set(std::size_t i) noexcept {
        slot_t mask{1ul << (i & bits_mask())};
        return InterlockedOr((long volatile*)&slots_[i / bits_per_slot()], mask) & mask;
    }

    inline void atomic_reset(std::size_t i) noexcept {
        slot_t mask{1ul << (i & bits_mask())};
        InterlockedAnd((long volatile*)&slots_[i / bits_per_slot()], ~mask);
    }

#else

    inline bool atomic_set(std::size_t i) noexcept {
        slot_t mask{1ul << (i & bits_mask())};
        return __atomic_fetch_or(&slots_[i / bits_per_slot()], mask, __ATOMIC_ACQUIRE) & mask;
    }

    inline void atomic_reset(std::size_t i) noexcept {
        slot_t mask{1ul << (i & bits_mask())};
        __atomic_fetch_and(&slots_[i / bits_per_slot()], ~mask, __ATOMIC_RELEASE);
    }

#endif
};

using visits_bitset_t = visits_bitset_gt<>;

/**
 *  @brief  Similar to `std::priority_queue`, but allows raw access to underlying
 *          memory, in case you want to shuffle it or sort. Good for collections
 *          from 100s to 10'000s elements.
 */
template <typename element_at,                                //
          typename comparator_at = std::less<void>,           // <void> is needed before C++14.
          typename allocator_at = std::allocator<element_at>> //
class max_heap_gt {
  public:
    using element_t = element_at;
    using comparator_t = comparator_at;
    using allocator_t = allocator_at;

    using value_type = element_t;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

  private:
    element_t* elements_;
    std::size_t size_;
    std::size_t capacity_;

  public:
    max_heap_gt() noexcept : elements_(nullptr), size_(0), capacity_(0) {}

    max_heap_gt(max_heap_gt&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)) {}

    max_heap_gt& operator=(max_heap_gt&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        return *this;
    }

    max_heap_gt(max_heap_gt const&) = delete;
    max_heap_gt& operator=(max_heap_gt const&) = delete;

    ~max_heap_gt() noexcept { reset(); }

    void reset() noexcept {
        if (elements_)
            allocator_t{}.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
        size_ = 0;
    }

    inline bool empty() const noexcept { return !size_; }
    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t capacity() const noexcept { return capacity_; }
    /// @brief  Selects the largest element in the heap.
    /// @return Reference to the stored element.
    inline element_t const& top() const noexcept { return elements_[0]; }
    inline void clear() noexcept { size_ = 0; }

    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity < capacity_)
            return true;

        new_capacity = ceil2(new_capacity);
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto allocator = allocator_t{};
        auto new_elements = allocator.allocate(new_capacity);
        if (!new_elements)
            return false;

        if (elements_) {
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
            allocator.deallocate(elements_, capacity_);
        }
        elements_ = new_elements;
        capacity_ = new_capacity;
        return new_elements;
    }

    bool insert(element_t&& element) noexcept {
        if (!reserve(size_ + 1))
            return false;

        insert_reserved(std::move(element));
        return true;
    }

    inline void insert_reserved(element_t&& element) noexcept {
        new (&elements_[size_]) element_t(element);
        size_++;
        shift_up(size_ - 1);
    }

    inline element_t pop() noexcept {
        element_t result = top();
        std::swap(elements_[0], elements_[size_ - 1]);
        size_--;
        elements_[size_].~element_t();
        shift_down(0);
        return result;
    }

    /** @brief Invalidates the "max-heap" property, transforming into ascending range. */
    inline void sort_ascending() noexcept { std::sort_heap(elements_, elements_ + size_, &less); }
    inline void shrink(std::size_t n) noexcept { size_ = (std::min<std::size_t>)(n, size_); }

    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }

  private:
    inline std::size_t parent_idx(std::size_t i) const noexcept { return (i - 1u) / 2u; }
    inline std::size_t left_child_idx(std::size_t i) const noexcept { return (i * 2u) + 1u; }
    inline std::size_t right_child_idx(std::size_t i) const noexcept { return (i * 2u) + 2u; }
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t{}(a, b); }

    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    void shift_down(std::size_t i) noexcept {
        std::size_t max_idx = i;

        std::size_t left = left_child_idx(i);
        if (left < size_ && less(elements_[max_idx], elements_[left]))
            max_idx = left;

        std::size_t right = right_child_idx(i);
        if (right < size_ && less(elements_[max_idx], elements_[right]))
            max_idx = right;

        if (i != max_idx) {
            std::swap(elements_[i], elements_[max_idx]);
            shift_down(max_idx);
        }
    }
};

/**
 *  @brief  Similar to `std::priority_queue`, but allows raw access to underlying
 *          memory and always keeps the data sorted. Ideal for small collections
 *          under 128 elements.
 */
template <typename element_at,                                //
          typename comparator_at = std::less<void>,           // <void> is needed before C++14.
          typename allocator_at = std::allocator<element_at>> //
class sorted_buffer_gt {
  public:
    using element_t = element_at;
    using comparator_t = comparator_at;
    using allocator_t = allocator_at;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

    using value_type = element_t;

  private:
    element_t* elements_;
    std::size_t size_;
    std::size_t capacity_;

  public:
    sorted_buffer_gt() noexcept : elements_(nullptr), size_(0), capacity_(0) {}

    sorted_buffer_gt(sorted_buffer_gt&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)) {}

    sorted_buffer_gt& operator=(sorted_buffer_gt&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        return *this;
    }

    sorted_buffer_gt(sorted_buffer_gt const&) = delete;
    sorted_buffer_gt& operator=(sorted_buffer_gt const&) = delete;

    ~sorted_buffer_gt() noexcept { reset(); }

    void reset() noexcept {
        if (elements_)
            allocator_t{}.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
        size_ = 0;
    }

    inline bool empty() const noexcept { return !size_; }
    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t capacity() const noexcept { return capacity_; }
    inline element_t const& top() const noexcept { return elements_[size_ - 1]; }
    inline void clear() noexcept { size_ = 0; }

    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity < capacity_)
            return true;

        new_capacity = ceil2(new_capacity);
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto allocator = allocator_t{};
        auto new_elements = allocator.allocate(new_capacity);
        if (!new_elements)
            return false;

        if (size_)
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
        if (elements_)
            allocator.deallocate(elements_, capacity_);

        elements_ = new_elements;
        capacity_ = new_capacity;
        return true;
    }

    inline void insert_reserved(element_t&& element) noexcept {
        std::size_t slot = size_ ? std::lower_bound(elements_, elements_ + size_, element, &less) - elements_ : 0;
        std::size_t to_move = size_ - slot;
        element_t* source = elements_ + size_ - 1;
        for (; to_move; --to_move, --source)
            source[1] = source[0];
        elements_[slot] = element;
        size_++;
    }

    /**
     *  @return `true` if the entry was added, `false` if it wasn't relevant enough.
     */
    inline bool insert(element_t&& element, std::size_t limit) noexcept {
        std::size_t slot = size_ ? std::lower_bound(elements_, elements_ + size_, element, &less) - elements_ : 0;
        if (slot == limit)
            return false;
        std::size_t to_move = size_ - slot - (size_ == limit);
        element_t* source = elements_ + size_ - 1 - (size_ == limit);
        for (; to_move; --to_move, --source)
            source[1] = source[0];
        elements_[slot] = element;
        size_ += size_ != limit;
        return true;
    }

    inline element_t pop() noexcept {
        size_--;
        element_t result = elements_[size_];
        elements_[size_].~element_t();
        return result;
    }

    void sort_ascending() noexcept {}
    inline void shrink(std::size_t n) noexcept { size_ = (std::min<std::size_t>)(n, size_); }

    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }

  private:
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t{}(a, b); }
};

#if defined(USEARCH_DEFINED_WINDOWS)
#pragma pack(push, 1) // Pack struct elements on 1-byte alignment
#endif

/**
 *  @brief Five-byte integer type to address node clouds with over 4B entries.
 *
 * @note Avoid usage in 32bit environment
 */
class usearch_pack_m uint40_t {
    unsigned char octets[5];

    inline uint40_t& broadcast(unsigned char c) {
        std::memset(octets, c, 5);
        return *this;
    }

  public:
    inline uint40_t() noexcept { broadcast(0); }
    inline uint40_t(std::uint32_t n) noexcept { std::memcpy(octets, (char*)&n, 4); }
    inline uint40_t(std::uint64_t n) noexcept { std::memcpy(octets, (char*)&n, 5); }
#if defined(USEARCH_DEFINED_CLANG) && defined(USEARCH_DEFINED_APPLE)
    inline uint40_t(std::size_t n) noexcept { std::memcpy(octets, (char*)&n, 5); }
#endif

    uint40_t(uint40_t&&) = default;
    uint40_t(uint40_t const&) = default;
    uint40_t& operator=(uint40_t&&) = default;
    uint40_t& operator=(uint40_t const&) = default;

    inline uint40_t& operator+=(std::uint32_t n) noexcept {
        std::uint32_t& tail = *reinterpret_cast<std::uint32_t*>(octets);
        octets[4] += static_cast<unsigned char>((tail + n) < tail);
        tail += n;
        return *this;
    }

    inline uint40_t& operator+=(std::size_t n) noexcept {
        unsigned char* n_octets = reinterpret_cast<unsigned char*>(&n);
        std::uint32_t& n_tail = *reinterpret_cast<std::uint32_t*>(n_octets);
        std::uint32_t& tail = *reinterpret_cast<std::uint32_t*>(octets);
        octets[4] += static_cast<unsigned char>((tail + n_tail) < tail);
        tail += n_tail;
        octets[4] += n_octets[4];
        return *this;
    }

    inline uint40_t operator+(std::size_t n) noexcept {
        uint40_t other(*this);
        other += n;
        return other;
    }

    inline operator std::size_t() const noexcept {
        std::size_t result = 0;
#ifdef USEARCH_64BIT_ENV
        std::memcpy((char*)&result + 3, octets, 5);
#else
        std::memcpy((char*)&result, octets + 1, 4);
#endif
        return result;
    }

    inline uint40_t& operator++() noexcept { return *this += 1u; }

    inline uint40_t operator++(int) noexcept {
        uint40_t old = *this;
        *this += 1u;
        return old;
    }

    inline static uint40_t max() noexcept { return uint40_t{}.broadcast(0xFF); }
    inline static uint40_t min() noexcept { return uint40_t{}.broadcast(0); }
};

#if defined(USEARCH_DEFINED_WINDOWS)
#pragma pack(pop) // Reset alignment to default
#endif

static_assert(sizeof(uint40_t) == 5, "uint40_t must be exactly 5 bytes");

/**
 *  @brief  Basic single-threaded @b ring class, used for all kinds of task queues.
 */
template <typename element_at, typename allocator_at = std::allocator<element_at>> //
class ring_gt {
  public:
    using element_t = element_at;
    using allocator_t = allocator_at;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

    using value_type = element_t;

  private:
    element_t* elements_{};
    std::size_t capacity_{};
    std::size_t head_{};
    std::size_t tail_{};
    bool empty_{true};
    allocator_t allocator_{};

  public:
    explicit ring_gt(allocator_t const& alloc = allocator_t()) noexcept : allocator_(alloc) {}

    ring_gt(ring_gt const&) = delete;
    ring_gt& operator=(ring_gt const&) = delete;

    ring_gt(ring_gt&& other) noexcept { swap(other); }
    ring_gt& operator=(ring_gt&& other) noexcept {
        swap(other);
        return *this;
    }

    void swap(ring_gt& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(capacity_, other.capacity_);
        std::swap(head_, other.head_);
        std::swap(tail_, other.tail_);
        std::swap(empty_, other.empty_);
        std::swap(allocator_, other.allocator_);
    }

    ~ring_gt() noexcept { reset(); }

    bool empty() const noexcept { return empty_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t size() const noexcept {
        if (empty_)
            return 0;
        else if (head_ >= tail_)
            return head_ - tail_;
        else
            return capacity_ - (tail_ - head_);
    }

    void clear() noexcept {
        head_ = 0;
        tail_ = 0;
        empty_ = true;
    }

    void reset() noexcept {
        if (elements_)
            allocator_.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
        head_ = 0;
        tail_ = 0;
        empty_ = true;
    }

    bool reserve(std::size_t n) noexcept {
        if (n < size())
            return false; // prevent data loss
        n = (std::max<std::size_t>)(n, 64u);
        element_t* elements = allocator_.allocate(n);
        if (!elements)
            return false;

        std::size_t i = 0;
        while (try_pop(elements[i]))
            i++;

        reset();
        elements_ = elements;
        capacity_ = n;
        head_ = i;
        tail_ = 0;
        empty_ = (i == 0);
        return true;
    }

    void push(element_t const& value) noexcept {
        elements_[head_] = value;
        head_ = (head_ + 1) % capacity_;
        empty_ = false;
    }

    bool try_push(element_t const& value) noexcept {
        if (head_ == tail_ && !empty_)
            return false; // elements_ is full

        return push(value);
        return true;
    }

    bool try_pop(element_t& value) noexcept {
        if (empty_)
            return false;

        value = std::move(elements_[tail_]);
        tail_ = (tail_ + 1) % capacity_;
        empty_ = head_ == tail_;
        return true;
    }

    element_t const& operator[](std::size_t i) const noexcept { return elements_[(tail_ + i) % capacity_]; }
};

/// @brief Number of neighbors per graph node.
/// Defaults to 32 in FAISS and 16 in hnswlib.
/// > It is called `M` in the paper.
constexpr std::size_t default_connectivity() { return 16; }

/// @brief Hyper-parameter controlling the quality of indexing.
/// Defaults to 40 in FAISS and 200 in hnswlib.
/// > It is called `efConstruction` in the paper.
constexpr std::size_t default_expansion_add() { return 128; }

/// @brief Hyper-parameter controlling the quality of search.
/// Defaults to 16 in FAISS and 10 in hnswlib.
/// > It is called `ef` in the paper.
constexpr std::size_t default_expansion_search() { return 64; }

constexpr std::size_t default_allocator_entry_bytes() { return 64; }

/**
 *  @brief  Configuration settings for the index construction.
 *          Includes the main `::connectivity` parameter (`M` in the paper)
 *          and two expansion factors - for construction and search.
 */
struct index_config_t {
    /// @brief Number of neighbors per graph node.
    /// Defaults to 32 in FAISS and 16 in hnswlib.
    /// > It is called `M` in the paper.
    std::size_t connectivity = default_connectivity();

    /// @brief Number of neighbors per graph node in base level graph.
    /// Defaults to double of the other levels, so 64 in FAISS and 32 in hnswlib.
    /// > It is called `M0` in the paper.
    std::size_t connectivity_base = default_connectivity() * 2;
};

struct index_limits_t {
    std::size_t members = 0;
    std::size_t threads_add = std::thread::hardware_concurrency();
    std::size_t threads_search = std::thread::hardware_concurrency();

    index_limits_t(std::size_t n, std::size_t t) noexcept : members(n), threads_add(t), threads_search(t) {}
    index_limits_t(std::size_t n = 0) noexcept : index_limits_t(n, std::thread::hardware_concurrency()) {}
    std::size_t threads() const noexcept { return (std::max)(threads_add, threads_search); }
    std::size_t concurrency() const noexcept { return (std::min)(threads_add, threads_search); }
};

struct add_config_t {
    /// @brief Hyper-parameter controlling the quality of indexing.
    /// Defaults to 40 in FAISS and 200 in hnswlib.
    /// > It is called `efConstruction` in the paper.
    std::size_t expansion = default_expansion_add();

    /// @brief Optional thread identifier for multi-threaded construction.
    std::size_t thread = 0;
};

struct search_config_t {
    /// @brief Hyper-parameter controlling the quality of search.
    /// Defaults to 16 in FAISS and 10 in hnswlib.
    /// > It is called `ef` in the paper.
    std::size_t expansion = default_expansion_search();

    /// @brief Optional thread identifier for multi-threaded construction.
    std::size_t thread = 0;

    /// @brief Brute-forces exhaustive search over all entries in the index.
    bool exact = false;
};

struct copy_config_t {};

struct join_config_t {
    /// @brief Controls maximum number of proposals per man during stable marriage.
    std::size_t max_proposals = 0;

    /// @brief Hyper-parameter controlling the quality of search.
    /// Defaults to 16 in FAISS and 10 in hnswlib.
    /// > It is called `ef` in the paper.
    std::size_t expansion = default_expansion_search();

    /// @brief Brute-forces exhaustive search over all entries in the index.
    bool exact = false;
};

/// @brief  C++17 and newer version deprecate the `std::result_of`
template <typename metric_at, typename... args_at>
using return_type_gt =
#if defined(USEARCH_DEFINED_CPP17)
    typename std::invoke_result<metric_at, args_at...>::type;
#else
    typename std::result_of<metric_at(args_at...)>::type;
#endif

struct dummy_predicate_t {
    template <typename match_at> constexpr bool operator()(match_at&&) const noexcept { return true; }
};

struct dummy_progress_t {
    inline void operator()(std::size_t /*progress*/, std::size_t /*total*/) const noexcept {}
};

struct dummy_executor_t {
    dummy_executor_t() noexcept {}
    std::size_t size() const noexcept { return 1; }

    template <typename thread_aware_function_at>
    void execute_bulk(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept {
        for (std::size_t task_idx = 0; task_idx != tasks; ++task_idx)
            thread_aware_function(0, task_idx);
    }

    template <typename thread_aware_function_at>
    void execute_bulk(thread_aware_function_at&& thread_aware_function) noexcept {
        thread_aware_function(0);
    }
};

struct dummy_label_to_label_mapping_t {
    struct member_ref_t {
        template <typename label_at> member_ref_t& operator=(label_at&&) noexcept { return *this; }
    };
    template <typename label_at> member_ref_t operator[](label_at&&) const noexcept { return {}; }
};

template <typename, typename at> struct has_reset_gt {
    static_assert(std::integral_constant<at, false>::value, "Second template parameter needs to be of function type.");
};

template <typename check_at, typename return_at, typename... args_at>
struct has_reset_gt<check_at, return_at(args_at...)> {
  private:
    template <typename at>
    static constexpr auto check(at*) ->
        typename std::is_same<decltype(std::declval<at>().reset(std::declval<args_at>()...)), return_at>::type;
    template <typename> static constexpr std::false_type check(...);

    typedef decltype(check<check_at>(0)) type;

  public:
    static constexpr bool value = type::value;
};

/**
 *  @brief  Checks if a certain class has a member function called `reset`.
 */
template <typename at> constexpr bool has_reset() { return has_reset_gt<at, void()>::value; }

struct serialization_result_t {
    error_t error;

    explicit operator bool() const noexcept { return !error; }
    serialization_result_t failed(error_t message) noexcept {
        error = std::move(message);
        return std::move(*this);
    }
};

class output_file_t {
    char const* path_ = nullptr;
    std::FILE* file_ = nullptr;

    void close() {
        if (file_)
            std::fclose(exchange(file_, nullptr));
    }

  public:
    output_file_t(char const* path) noexcept : path_(path) {}
    ~output_file_t() noexcept { close(); }
    output_file_t(output_file_t&& other) noexcept
        : path_(exchange(other.path_, nullptr)), file_(exchange(other.file_, nullptr)) {}
    output_file_t& operator=(output_file_t&& other) noexcept {
        std::swap(path_, other.path_);
        std::swap(file_, other.file_);
        return *this;
    }
    serialization_result_t open_if_not() noexcept {
        serialization_result_t result;
        if (file_)
            return result;
        file_ = std::fopen(path_, "wb");
        if (!file_)
            return result.failed(std::strerror(errno));
        return result;
    }
    serialization_result_t write(void* begin, std::size_t length) noexcept {
        serialization_result_t result;
        std::size_t written = std::fwrite(begin, length, 1, file_);
        if (!written) {
            std::fclose(file_);
            result.failed(std::strerror(errno));
        }
        return result;
    }
};

class input_file_t {
    char const* path_ = nullptr;
    std::FILE* file_ = nullptr;

    void close() {
        if (file_)
            std::fclose(exchange(file_, nullptr));
    }

  public:
    input_file_t(char const* path) noexcept : path_(path) {}
    ~input_file_t() noexcept { close(); }
    input_file_t(input_file_t&& other) noexcept
        : path_(exchange(other.path_, nullptr)), file_(exchange(other.file_, nullptr)) {}
    input_file_t& operator=(input_file_t&& other) noexcept {
        std::swap(path_, other.path_);
        std::swap(file_, other.file_);
        return *this;
    }

    serialization_result_t open_if_not() noexcept {
        serialization_result_t result;
        if (file_)
            return result;
        file_ = std::fopen(path_, "rb");
        if (!file_)
            return result.failed(std::strerror(errno));
        return result;
    }
    serialization_result_t read(void* begin, std::size_t length) noexcept {
        serialization_result_t result;
        std::size_t read = std::fread(begin, length, 1, file_);
        if (!read) {
            std::fclose(file_);
            result.failed(std::strerror(errno));
        }
        return result;
    }
};

/// @brief  OS-specific to wrap open file-descriptors/handles.
class memory_mapped_file_t {
    char const* path_{};
    void* ptr_{};
    size_t length_{};

#if defined(USEARCH_DEFINED_WINDOWS)
    HANDLE file_handle_{};
    HANDLE mapping_handle_{};
#else
    int file_descriptor_{};
#endif

  public:
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    byte_t const* data() const noexcept { return reinterpret_cast<byte_t const*>(ptr_); }
    std::size_t size() const noexcept { return static_cast<std::size_t>(length_); }

    memory_mapped_file_t() noexcept {}
    memory_mapped_file_t(char const* path) noexcept : path_(path) {}
    ~memory_mapped_file_t() noexcept { close(); }
    memory_mapped_file_t(memory_mapped_file_t&& other) noexcept
        : path_(exchange(other.path_, nullptr)), ptr_(exchange(other.ptr_, nullptr)),
          length_(exchange(other.length_, 0)),
#if defined(USEARCH_DEFINED_WINDOWS)
          file_handle_(exchange(other.file_handle_, nullptr)), mapping_handle_(exchange(other.mapping_handle_, nullptr))
#else
          file_descriptor_(exchange(other.file_descriptor_, 0))
#endif
    {
    }

    memory_mapped_file_t& operator=(memory_mapped_file_t&& other) noexcept {
        std::swap(path_, other.path_);
        std::swap(ptr_, other.ptr_);
        std::swap(length_, other.length_);
#if defined(USEARCH_DEFINED_WINDOWS)
        std::swap(file_handle_, other.file_handle_);
        std::swap(mapping_handle_, other.mapping_handle_);
#else
        std::swap(file_descriptor_, other.file_descriptor_);
#endif
        return *this;
    }

    serialization_result_t open_if_not() noexcept {
        serialization_result_t result;

#if defined(USEARCH_DEFINED_WINDOWS)

        HANDLE file_handle =
            CreateFile(path_, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        if (file_handle == INVALID_HANDLE_VALUE)
            return result.failed("Opening file failed!");

        file_offset_t file_length = GetFileSize(file_handle, 0);
        HANDLE mapping_handle = CreateFileMapping(file_handle, 0, PAGE_READONLY, 0, 0, 0);
        if (mapping_handle == 0) {
            CloseHandle(file_handle);
            return result.failed("Mapping file failed!");
        }

        byte_t* file = (byte_t*)MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, file_length);
        if (file == 0) {
            CloseHandle(mapping_handle);
            CloseHandle(file_handle);
            return result.failed("View the map failed!");
        }
        file_handle_ = file_handle;
        mapping_handle_ = mapping_handle;
        ptr_ = file;
        length_ = file_length;
#else

#if defined(USEARCH_DEFINED_LINUX)
        int descriptor = open(path_, O_RDONLY | O_NOATIME);
#else
        int descriptor = open(path_, O_RDONLY);
#endif

        // Estimate the file size
        struct stat file_stat;
        int fstat_status = fstat(descriptor, &file_stat);
        if (fstat_status < 0) {
            ::close(descriptor);
            return result.failed(std::strerror(errno));
        }

        // Map the entire file
        byte_t* file = (byte_t*)mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (file == MAP_FAILED) {
            ::close(descriptor);
            return result.failed(std::strerror(errno));
        }
        file_descriptor_ = descriptor;
        ptr_ = file;
        length_ = file_stat.st_size;
#endif // Platform specific code
        return result;
    }

    void close() noexcept {
        if (!ptr_)
            return;
#if defined(USEARCH_DEFINED_WINDOWS)
        UnmapViewOfFile(ptr_);
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
#else
        munmap(ptr_, length_);
        ::close(file_descriptor_);
        file_descriptor_ = 0;
#endif
        ptr_ = nullptr;
        length_ = 0;
    }
};

struct index_serialized_header_t {
    std::uint64_t size = 0;
    std::uint64_t connectivity = 0;
    std::uint64_t connectivity_base = 0;
    std::uint64_t max_level = 0;
    std::uint64_t entry_idx = 0;
};

using default_label_t = std::uint64_t;
using default_internal_id_t = std::uint32_t;

/**
 *  @brief  Approximate Nearest Neighbors Search @b index-structure using the
 *          Hierarchical Navigable Small World @b (HNSW) graphs algorithm.
 *
 *  @section Features
 *      - Thread-safe for concurrent construction, search, and updates.
 *      - Doesn't allocate new threads, and reuses the ones its called from.
 *      - Allows storing content externally, managing just the similarity index.
 *      - Joins.
 *
 *  @tparam metric_at
 *      A function responsible for computing the distance (dis-similarity) between two vectors.
 *      Can have following signatures:
 *          - `distance_t (*) (content, member_cref_t)`
 *          - `distance_t (*) (member_cref_t, member_cref_t)`
 *      Alternatively, may implement the batch-search capability:
 *          - `void batch(content, members)`
 *
 *  @tparam label_at
 *      The type of unique labels to assign to vectors.
 *
 *  @tparam internal_id_at
 *      The smallest unsigned integer type to address indexed elements.
 *      Can be a built-in `uint32_t`, `uint64_t`, or our custom `uint40_t`.
 *
 *  @tparam dynamic_allocator_at
 *      Dynamic memory allocator for temporary buffers, visits indicators, and
 *      priority queues, needed during construction and traversals of graphs.
 *      The allocated buffers may be uninitialized.
 *
 *  @tparam tape_allocator_at
 *      Potentially different memory allocator for primary allocations of nodes and vectors.
 *      It would never `deallocate` separate entries, and would only free all the space at once.
 *      The allocated buffers may be uninitialized.
 *
 *  @section Usage
 *
 *  @subsection Exceptions
 *
 *  None of the methods throw exceptions in the "Release" compilation mode.
 *  It may only `throw` if your memory ::dynamic_allocator_at or ::metric_at isn't
 *  safe to copy.
 *
 *  @subsection Serialization
 *
 *  When serialized, doesn't include any additional metadata.
 *  It is just the multi-level proximity-graph. You may want to store metadata about
 *  the used metric and label types somewhere else.
 *
 *  @section Implementation Details
 *
 *  Like every HNSW implementation, USearch builds levels of "Proximity Graphs".
 *  Every added vector forms a node in one or more levels of the graph.
 *  Every node is present in the base level. Every following level contains a smaller
 *  fraction of nodes. During search, the operation starts with the smaller levels
 *  and zooms-in on every following iteration of larger graph traversals.
 *
 *  Just one memory allocation is performed regardless of the number of levels.
 *  The adjacency lists across all levels are concatenated into that single buffer.
 *  That buffer starts with a "head", that stores the metadata, such as the
 *  tallest "level" of the graph that it belongs to, the external "label", and the
 *  number of "dimensions" in the vector.
 *
 */
template <typename metric_at,                                     //
          typename label_at = default_label_t,                    //
          typename internal_id_at = default_internal_id_t,        //
          typename dynamic_allocator_at = std::allocator<byte_t>, //
          typename tape_allocator_at = dynamic_allocator_at>      //
class index_gt {
  public:
    using metric_t = metric_at;
    using label_t = label_at;
    using internal_id_t = internal_id_at;
    using dynamic_allocator_t = dynamic_allocator_at;
    using tape_allocator_t = tape_allocator_at;
    static_assert(sizeof(label_t) >= sizeof(internal_id_t), "Having tiny labels doesn't make sense.");

    /**
     *  @brief  Scalar describing distance between two members.
     *          It is inferred from the passed ::metric_t, as the result type
     *          of calling its `operator () (label_t, label_t)`.
     */
    using distance_t = return_type_gt<metric_t, label_t, label_t>;

    struct member_cref_t {
        label_t label;
        internal_id_t id;
    };

    struct member_ref_t {
        misaligned_ref_gt<label_t> label;
        internal_id_t id;

        inline operator member_cref_t() const noexcept { return {label, id}; }
    };

    template <typename ref_at, typename index_at> class member_iterator_gt {
        using ref_t = ref_at;
        using index_t = index_at;
        index_t* index_{};
        std::size_t offset_{};

        friend class index_gt;
        member_iterator_gt() noexcept {}
        member_iterator_gt(index_t* index, std::size_t offset) noexcept : index_(index), offset_(offset) {}

      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = ref_t;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = ref_t;

        reference operator*() const noexcept {
            return {index_->node_at_(offset_).label(), static_cast<internal_id_t>(offset_)};
        }

        member_iterator_gt operator++(int) noexcept { return member_iterator_gt(index_, offset_ + 1); }
        member_iterator_gt operator--(int) noexcept { return member_iterator_gt(index_, offset_ - 1); }
        member_iterator_gt operator+(difference_type d) noexcept { return member_iterator_gt(index_, offset_ + d); }
        member_iterator_gt operator-(difference_type d) noexcept { return member_iterator_gt(index_, offset_ - d); }

        member_iterator_gt& operator++() noexcept {
            offset_ += 1;
            return *this;
        }
        member_iterator_gt& operator--() noexcept {
            offset_ -= 1;
            return *this;
        }
        member_iterator_gt& operator+=(difference_type d) noexcept {
            offset_ += d;
            return *this;
        }
        member_iterator_gt& operator-=(difference_type d) noexcept {
            offset_ -= d;
            return *this;
        }
        bool operator==(member_iterator_gt const& other) const noexcept {
            return index_ == other.index_ && offset_ == other.offset_;
        }
        bool operator!=(member_iterator_gt const& other) const noexcept {
            return index_ != other.index_ || offset_ != other.offset_;
        }
    };

    using member_iterator_t = member_iterator_gt<member_ref_t, index_gt>;
    using member_citerator_t = member_iterator_gt<member_cref_t, index_gt const>;

    // STL compatibility:
    using value_type = std::pair<label_t, distance_t>;
    using allocator_type = dynamic_allocator_t;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = member_ref_t;
    using const_reference = member_cref_t;
    using pointer = void;
    using const_pointer = void;
    using iterator = member_iterator_t;
    using const_iterator = member_citerator_t;
    using reverse_iterator = std::reverse_iterator<member_iterator_t>;
    using reverse_const_iterator = std::reverse_iterator<member_citerator_t>;

  private:
    using neighbors_count_t = std::uint32_t;
    using level_t = std::int32_t;

    using allocator_traits_t = std::allocator_traits<dynamic_allocator_t>;
    using byte_t = typename dynamic_allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, //
                  "Primary allocator must allocate separate addressable bytes");

    using tape_allocator_traits_t = std::allocator_traits<tape_allocator_t>;
    static_assert(sizeof(typename tape_allocator_traits_t::value_type) == 1, //
                  "Tape allocator must allocate separate addressable bytes");

    /**
     *  @brief  How many bytes of memory are needed to form the "head" of the node.
     */
    static constexpr std::size_t node_head_bytes_() { return sizeof(label_t) + sizeof(level_t); }

    using visits_bitset_t = visits_bitset_gt<dynamic_allocator_t>;

    struct precomputed_constants_t {
        double inverse_log_connectivity{};
        std::size_t neighbors_bytes{};
        std::size_t neighbors_base_bytes{};
    };
    struct candidate_t {
        distance_t distance;
        internal_id_t id;
    };
    struct compare_by_distance_t {
        inline bool operator()(candidate_t a, candidate_t b) const noexcept { return a.distance < b.distance; }
    };

    using candidates_view_t = span_gt<candidate_t const>;
    using candidates_allocator_t = typename allocator_traits_t::template rebind_alloc<candidate_t>;
    using top_candidates_t = sorted_buffer_gt<candidate_t, compare_by_distance_t, candidates_allocator_t>;
    using next_candidates_t = max_heap_gt<candidate_t, compare_by_distance_t, candidates_allocator_t>;

    /**
     *  @brief  A loosely-structured handle for every node. One such node is created for every member.
     *          To minimize memory usage and maximize the number of entries per cache-line, it only
     *          stores to pointers. The internal tape starts with a `label_t` @b label, then
     *          a `level_t` for the number of graph @b levels in which this member appears,
     *          then the { `neighbors_count_t`, `internal_id_t`, `internal_id_t` ... } sequences
     *          for @b each-level.
     */
    class node_t {
        byte_t* tape_{};

      public:
        explicit node_t(byte_t* tape) noexcept : tape_(tape) {}
        byte_t* tape() const noexcept { return tape_; }
        byte_t* neighbors_tape() const noexcept { return tape_ + node_head_bytes_(); }
        explicit operator bool() const noexcept { return tape_; }

        node_t() = default;
        node_t(node_t const&) = default;
        node_t& operator=(node_t const&) = default;

        misaligned_ref_gt<label_t> label() const noexcept { return {tape_}; }
        misaligned_ref_gt<level_t> level() const noexcept { return {tape_ + sizeof(label_t)}; }

        void label(label_t v) noexcept { return misaligned_store<label_t>(tape_, v); }
        void level(level_t v) noexcept { return misaligned_store<level_t>(tape_ + sizeof(label_t), v); }
    };

    /**
     *  @brief  A slice of the node's tape, containing a the list of neighbors
     *          for a node in a single graph level. It's pre-allocated to fit
     *          as many neighbors IDs, as may be needed at the target level,
     *          and starts with a single integer `neighbors_count_t` counter.
     */
    class neighbors_ref_t {
        byte_t* tape_;

        static constexpr std::size_t shift(std::size_t i = 0) {
            return sizeof(neighbors_count_t) + sizeof(internal_id_t) * i;
        }

      public:
        neighbors_ref_t(byte_t* tape) noexcept : tape_(tape) {}
        misaligned_ptr_gt<internal_id_t> begin() noexcept { return tape_ + shift(); }
        misaligned_ptr_gt<internal_id_t> end() noexcept { return begin() + size(); }
        misaligned_ptr_gt<internal_id_t const> begin() const noexcept { return tape_ + shift(); }
        misaligned_ptr_gt<internal_id_t const> end() const noexcept { return begin() + size(); }
        internal_id_t operator[](std::size_t i) const noexcept {
            return misaligned_load<internal_id_t>(tape_ + shift(i));
        }
        std::size_t size() const noexcept { return misaligned_load<neighbors_count_t>(tape_); }
        void clear() noexcept { misaligned_store<neighbors_count_t>(tape_, 0); }
        void push_back(internal_id_t id) noexcept {
            neighbors_count_t n = misaligned_load<neighbors_count_t>(tape_);
            misaligned_store<internal_id_t>(tape_ + shift(n), id);
            misaligned_store<neighbors_count_t>(tape_, n + 1);
        }
    };

    /**
     *  @brief  A package of all kinds of temporary data-structures, that the threads
     *          would reuse to process requests. Similar to having all of those as
     *          separate `thread_local` global variables.
     */
    struct usearch_align_m context_t {
        top_candidates_t top_candidates{};
        next_candidates_t next_candidates{};
        visits_bitset_t visits{};
        std::default_random_engine level_generator{};
        metric_t metric{};
        std::size_t iteration_cycles{};
        std::size_t measurements_count{};

        template <typename query_at> inline distance_t measure(query_at&& first, node_t second) noexcept {
            measurements_count++;
            return metric(std::forward<query_at>(first), second.label());
        }

        inline distance_t measure(node_t first, node_t second) noexcept {
            measurements_count++;
            return metric(first.label(), second.label());
        }
    };

    index_config_t config_{};
    index_limits_t limits_{};
    metric_t metric_{};
    mutable dynamic_allocator_t dynamic_allocator_{};
    tape_allocator_t tape_allocator_{};
    precomputed_constants_t pre_{};
    memory_mapped_file_t viewed_file_{};

    usearch_align_m mutable std::atomic<std::size_t> capacity_{};
    usearch_align_m mutable std::atomic<std::size_t> size_{};

    /// @brief  Controls access to `max_level_` and `entry_idx_`.
    ///         If any thread is updating those values, no other threads can `add()` or `search()`.
    std::mutex global_mutex_{};
    level_t max_level_{};
    internal_id_t entry_idx_{};

    using nodes_allocator_t = typename allocator_traits_t::template rebind_alloc<node_t>;
    node_t* nodes_{};
    mutable visits_bitset_t nodes_mutexes_{};

    using contexts_allocator_t = typename allocator_traits_t::template rebind_alloc<context_t>;
    context_t* contexts_{};

  public:
    std::size_t connectivity() const noexcept { return config_.connectivity; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t max_level() const noexcept { return static_cast<std::size_t>(max_level_); }
    index_config_t const& config() const noexcept { return config_; }
    index_limits_t const& limits() const noexcept { return limits_; }
    bool is_immutable() const noexcept { return bool(viewed_file_); }

    /**
     *  @section Exceptions
     *      Doesn't throw, unless the ::metric's and ::allocators's throw on copy-construction.
     */
    explicit index_gt(index_config_t config = {}, metric_t metric = {}, dynamic_allocator_t allocator = {},
                      tape_allocator_t tape_allocator = {}) noexcept
        : config_(config), limits_(0, 0), metric_(metric), dynamic_allocator_(std::move(allocator)),
          tape_allocator_(std::move(tape_allocator)), pre_(precompute_(config)), size_(0u), max_level_(-1),
          entry_idx_(0u), nodes_(nullptr), nodes_mutexes_(), contexts_(nullptr) {}

    /**
     *  @brief  Clones the structure with the same hyper-parameters, but without contents.
     */
    index_gt fork() noexcept { return index_gt{config_, metric_, dynamic_allocator_}; }

    ~index_gt() noexcept { reset(); }

    index_gt(index_gt&& other) noexcept { swap(other); }

    index_gt& operator=(index_gt&& other) noexcept {
        swap(other);
        return *this;
    }

    struct copy_result_t {
        error_t error;
        index_gt index;

        explicit operator bool() const noexcept { return !error; }
        copy_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    copy_result_t copy(copy_config_t /*config*/ = {}) const noexcept {
        copy_result_t result;
        index_gt& other = result.index;
        other = index_gt(config_, metric_, dynamic_allocator_, tape_allocator_);
        if (!other.reserve(limits_))
            return result.failed("Failed to reserve the contexts");

        // Now all is left - is to allocate new `node_t` instances and populate
        // the `other.nodes_` array into it.
        for (std::size_t i = 0; i != size_; ++i)
            other.nodes_[i] = other.node_make_copy_(node_bytes_(nodes_[i]));

        other.size_ = size_.load();
        other.max_level_ = max_level_;
        other.entry_idx_ = entry_idx_;
        return result;
    }

    member_citerator_t cbegin() const noexcept { return {this, 0}; }
    member_citerator_t cend() const noexcept { return {this, size()}; }
    member_citerator_t begin() const noexcept { return {this, 0}; }
    member_citerator_t end() const noexcept { return {this, size()}; }
    member_iterator_t begin() noexcept { return {this, 0}; }
    member_iterator_t end() noexcept { return {this, size()}; }
    member_ref_t at(std::size_t i) noexcept { return {nodes_[i].label(), static_cast<internal_id_t>(i)}; }
    member_cref_t at(std::size_t i) const noexcept { return {nodes_[i].label(), static_cast<internal_id_t>(i)}; }

    dynamic_allocator_t const& dynamic_allocator() const noexcept { return dynamic_allocator_; }
    tape_allocator_t const& tape_allocator() const noexcept { return tape_allocator_; }

#pragma region Adjusting Configuration

    /**
     *  @brief  Erases all the vectors from the index.
     *          Will change `size()` to zero, but will keep the same `capacity()`.
     *          Will keep the number of available threads/contexts the same as it was.
     */
    void clear() noexcept {
        if (!has_reset<tape_allocator_t>()) {
            std::size_t n = size_;
            for (std::size_t i = 0; i != n; ++i)
                node_free_(i);
        } else
            tape_allocator_.deallocate(nullptr, 0);
        size_ = 0;
        max_level_ = -1;
        entry_idx_ = 0u;
    }

    /**
     *  @brief  Erases all the vectors from the index, also deallocating the registry.
     *          Will change both `size()` and `capacity()` to zero.
     *          Will deallocate all threads/contexts.
     */
    void reset() noexcept {
        clear();

        if (nodes_)
            nodes_allocator_t{}.deallocate(exchange(nodes_, nullptr), limits_.members);
        if (contexts_) {
            for (std::size_t i = 0; i != limits_.threads(); ++i)
                contexts_[i].~context_t();
            contexts_allocator_t{}.deallocate(exchange(contexts_, nullptr), limits_.threads());
        }
        limits_ = index_limits_t{0, 0};
        capacity_ = 0;
        viewed_file_ = memory_mapped_file_t{};
    }

    /**
     *  @brief  Swaps the underlying memory buffers and thread contexts.
     */
    void swap(index_gt& other) noexcept {
        std::swap(config_, other.config_);
        std::swap(limits_, other.limits_);
        std::swap(metric_, other.metric_);
        std::swap(dynamic_allocator_, other.dynamic_allocator_);
        std::swap(tape_allocator_, other.tape_allocator_);
        std::swap(pre_, other.pre_);
        std::swap(viewed_file_, other.viewed_file_);
        std::swap(max_level_, other.max_level_);
        std::swap(entry_idx_, other.entry_idx_);
        std::swap(nodes_, other.nodes_);
        std::swap(nodes_mutexes_, other.nodes_mutexes_);
        std::swap(contexts_, other.contexts_);

        // Non-atomic parts.
        std::size_t capacity_copy = capacity_;
        std::size_t size_copy = size_;
        capacity_ = other.capacity_.load();
        size_ = other.size_.load();
        other.capacity_ = capacity_copy;
        other.size_ = size_copy;
    }

    /**
     *  @brief  Increases the `capacity()` of the index to allow adding more vectors.
     *  @return `true` on success, `false` on memory allocation errors.
     */
    bool reserve(index_limits_t limits) usearch_noexcept_m {

        if (limits.threads_add <= limits_.threads_add          //
            && limits.threads_search <= limits_.threads_search //
            && limits.members <= limits_.members)
            return true;

        if (!nodes_mutexes_.resize(limits.members))
            return false;

        nodes_allocator_t node_allocator;
        node_t* new_nodes = node_allocator.allocate(limits.members);
        if (!new_nodes)
            return false;

        std::size_t limits_threads = limits.threads();
        contexts_allocator_t context_allocator;
        context_t* new_contexts = context_allocator.allocate(limits_threads);
        if (!new_contexts) {
            node_allocator.deallocate(new_nodes, limits.members);
            return false;
        }
        for (std::size_t i = 0; i != limits_threads; ++i) {
            context_t& context = new_contexts[i];
            new (&context) context_t();
            context.metric = metric_;
            if (!context.visits.resize(limits.members)) {
                for (std::size_t j = 0; j != i; ++j)
                    context.visits.reset();
                node_allocator.deallocate(new_nodes, limits.members);
                context_allocator.deallocate(new_contexts, limits_threads);
                return false;
            }
        }

        // We have passed all the require memory allocations.
        // The remaining code can't fail. Let's just reuse some of our existing buffers.
        for (std::size_t i = 0; i != limits_.threads(); ++i) {
            context_t& old_context = contexts_[i];
            context_t& context = new_contexts[i];
            std::swap(old_context.top_candidates, context.top_candidates);
            std::swap(old_context.next_candidates, context.next_candidates);
            std::swap(old_context.iteration_cycles, context.iteration_cycles);
            std::swap(old_context.measurements_count, context.measurements_count);
            old_context.visits.reset();
        }

        // Move the nodes info, and deallocate previous buffers.
        if (nodes_)
            std::memcpy(new_nodes, nodes_, sizeof(node_t) * size()), node_allocator.deallocate(nodes_, limits_.members);
        if (contexts_)
            context_allocator.deallocate(contexts_, limits_.threads());

        limits_ = limits;
        capacity_ = limits.members;
        nodes_ = new_nodes;
        contexts_ = new_contexts;
        return true;
    }

#pragma endregion

#pragma region Construction and Search

    struct add_result_t {
        error_t error{};
        std::size_t new_size{};
        std::size_t cycles{};
        std::size_t measurements{};
        internal_id_t id{};

        explicit operator bool() const noexcept { return !error; }
        add_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    struct match_t {
        member_cref_t member;
        distance_t distance;
    };

    class search_result_t {
        index_gt const& index_;
        top_candidates_t& top_;

        friend class index_gt;
        inline search_result_t(index_gt const& index, top_candidates_t& top) noexcept : index_(index), top_(top) {}

      public:
        std::size_t count{};
        std::size_t cycles{};
        std::size_t measurements{};
        error_t error{};

        // todo: this is a hack to get a handle to a failed search_result_t without changing its member types
        // given that search_result_t can represent a failed result (which implies no valid top_candidates_t &top_),
        // the top_ probably needs to be a pointer.
        inline search_result_t(index_gt const& index) noexcept
            : index_(index), top_(index.contexts_[0].top_candidates) {}
        inline search_result_t(search_result_t&&) = default;
        inline search_result_t& operator=(search_result_t&&) = default;

        explicit operator bool() const noexcept { return !error; }
        search_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }

        inline operator std::size_t() const noexcept { return count; }
        inline std::size_t size() const noexcept { return count; }
        inline bool empty() const noexcept { return !count; }
        inline match_t operator[](std::size_t i) const noexcept { return at(i); }
        inline match_t front() const noexcept { return at(0); }
        inline match_t back() const noexcept { return at(count - 1); }
        inline bool contains(label_t label) const noexcept {
            for (std::size_t i = 0; i != count; ++i)
                if (at(i).member.label == label)
                    return true;
            return false;
        }
        inline match_t at(std::size_t i) const noexcept {
            candidate_t const* top_ordered = top_.data();
            candidate_t candidate = top_ordered[i];
            node_t node = index_.node_at_(candidate.id);
            return {member_cref_t{node.label(), candidate.id}, candidate.distance};
        }
        inline std::size_t merge_into(              //
            label_t* labels, distance_t* distances, //
            std::size_t old_count, std::size_t max_count) const noexcept {

            std::size_t merged_count = old_count;
            for (std::size_t i = 0; i != count; ++i) {
                match_t result = operator[](i);
                auto merged_end = distances + merged_count;
                auto offset = std::lower_bound(distances, merged_end, result.distance) - distances;
                if (offset == max_count)
                    continue;

                std::size_t count_worse = merged_count - offset - (max_count == merged_count);
                std::memmove(labels + offset + 1, labels + offset, count_worse * sizeof(label_t));
                std::memmove(distances + offset + 1, distances + offset, count_worse * sizeof(distance_t));
                labels[merged_count] = result.member.label;
                distances[merged_count] = result.distance;
                merged_count += 1;
            }
            return merged_count;
        }
        inline std::size_t dump_to(label_t* labels, distance_t* distances) const noexcept {
            for (std::size_t i = 0; i != count; ++i) {
                match_t result = operator[](i);
                labels[i] = result.member.label;
                distances[i] = result.distance;
            }
            return count;
        }
        inline std::size_t dump_to(label_t* labels) const noexcept {
            for (std::size_t i = 0; i != count; ++i) {
                match_t result = operator[](i);
                labels[i] = result.member.label;
            }
            return count;
        }
    };

    /**
     *  @brief Inserts a new entry into the index. Thread-safe.
     *
     *  @param[in] label External identifier/name/descriptor for the new entry.
     *  @param[in] content Content that will be compared against other entries to index.
     *  @param[in] config Configuration options for this specific operation.
     */
    template <typename content_at>
    add_result_t add(label_t label, content_at&& content, add_config_t config = {}) usearch_noexcept_m {

        usearch_assert_m(!is_immutable(), "Can't add to an immutable index");
        add_result_t result;

        // Make sure we have enough local memory to perform this request
        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        next_candidates_t& next = context.next_candidates;
        top.clear();
        next.clear();

        // The top list needs one more slot than the connectivity of the base level
        // for the heuristic, that tries to squeeze one more element into saturated list.
        std::size_t top_limit = (std::max)(config_.connectivity_base + 1, config.expansion);
        if (!top.reserve(top_limit))
            return result.failed("Out of memory!");
        if (!next.reserve(config.expansion))
            return result.failed("Out of memory!");

        // Determining how much memory to allocate for the node depends on the target level
        std::unique_lock<std::mutex> new_level_lock(global_mutex_);
        level_t max_level_copy = max_level_;       // Copy under lock
        internal_id_t entry_idx_copy = entry_idx_; // Copy under lock
        level_t target_level = choose_random_level_(context.level_generator);
        if (target_level <= max_level_copy)
            new_level_lock.unlock();

        // Allocate the neighbors
        node_t node = node_make_(label, target_level);
        if (!node)
            return result.failed("Out of memory!");
        std::size_t old_size = size_.fetch_add(1);
        internal_id_t new_id = static_cast<internal_id_t>(old_size);
        nodes_[old_size] = node;
        result.new_size = old_size + 1;
        result.id = new_id;
        node_lock_t new_lock = node_lock_(old_size);

        // Do nothing for the first element
        if (!new_id) {
            entry_idx_ = new_id;
            max_level_ = target_level;
            return result;
        }

        // Pull stats
        result.measurements = context.measurements_count;
        result.cycles = context.iteration_cycles;

        connect_node_across_levels_(new_id, content, entry_idx_copy, max_level_copy, target_level, config, context);

        // Normalize stats
        result.measurements = context.measurements_count - result.measurements;
        result.cycles = context.iteration_cycles - result.cycles;

        // Updating the entry point if needed
        if (target_level > max_level_copy) {
            entry_idx_ = new_id;
            max_level_ = target_level;
        }
        return result;
    }

    /**
     *  @brief Update an existing entry. Thread-safe.
     *
     *  @param[in] old_id Existing internal identifier for a node to be replaced.
     *  @param[in] label External identifier/name/descriptor for the entry.
     *  @param[in] content Content that will be compared against other entries in the index.
     *  @param[in] config Configuration options for this specific operation.
     */
    template <typename content_at>
    add_result_t update(internal_id_t old_id, label_t label, content_at&& content,
                        add_config_t config = {}) usearch_noexcept_m {

        usearch_assert_m(!is_immutable(), "Can't add to an immutable index");
        add_result_t result;

        // Make sure we have enough local memory to perform this request
        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        next_candidates_t& next = context.next_candidates;
        top.clear();
        next.clear();

        // The top list needs one more slot than the connectivity of the base level
        // for the heuristic, that tries to squeeze one more element into saturated list.
        std::size_t top_limit = (std::max)(config_.connectivity_base + 1, config.expansion);
        if (!top.reserve(top_limit))
            return result.failed("Out of memory!");
        if (!next.reserve(config.expansion))
            return result.failed("Out of memory!");

        node_lock_t new_lock = node_lock_(old_id);
        node_t node = node_at_(old_id);

        // Pull stats
        result.measurements = context.measurements_count;
        result.cycles = context.iteration_cycles;

        connect_node_across_levels_(old_id, content, entry_idx_, max_level_, node.level(), config, context);
        node.label(label);

        // Normalize stats
        result.measurements = context.measurements_count - result.measurements;
        result.cycles = context.iteration_cycles - result.cycles;

        return result;
    }

    /**
     *  @brief Searches for the closest elements to the given ::query. Thread-safe.
     *
     *  @param[in] query Content that will be compared against other entries in the index.
     *  @param[in] wanted The upper bound for the number of results to return.
     *  @param[in] config Configuration options for this specific operation.
     *  @param[in] predicate Optional filtering predicate for `member_cref_t`.
     *  @return Smart object referencing temporary memory. Valid until next `search()` or `add()`.
     */
    template <typename query_at, typename predicate_at = dummy_predicate_t>
    search_result_t search( //
        query_at query, std::size_t wanted, search_config_t config = {},
        predicate_at&& predicate = predicate_at{}) const noexcept {

        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        search_result_t result{*this, top};
        if (!size_)
            return result;

        // Go down the level, tracking only the closest match
        result.measurements = context.measurements_count;
        result.cycles = context.iteration_cycles;

        if (config.exact) {
            if (!top.reserve(wanted))
                return result.failed("Out of memory!");
            search_exact_(query, wanted, context, std::forward<predicate_at>(predicate));
        } else {
            next_candidates_t& next = context.next_candidates;
            std::size_t expansion = (std::max)(config.expansion, wanted);
            if (!next.reserve(expansion))
                return result.failed("Out of memory!");
            if (!top.reserve(expansion))
                return result.failed("Out of memory!");

            internal_id_t closest_id = search_for_one_(entry_idx_, query, max_level_, 0, context);
            // For bottom layer we need a more optimized procedure
            if (!search_to_find_in_base_(closest_id, query, expansion, context, std::forward<predicate_at>(predicate)))
                return result.failed("Out of memory!");
        }

        top.sort_ascending();
        top.shrink(wanted);

        // Normalize stats
        result.measurements = context.measurements_count - result.measurements;
        result.cycles = context.iteration_cycles - result.cycles;
        result.count = top.size();
        return result;
    }

#pragma endregion

#pragma region Metadata

    struct stats_t {
        std::size_t nodes;
        std::size_t edges;
        std::size_t max_edges;
        std::size_t allocated_bytes;
    };

    stats_t stats() const noexcept {
        stats_t result{};
        result.nodes = size();
        for (std::size_t i = 0; i != result.nodes; ++i) {
            node_t node = node_at_(i);
            std::size_t max_edges = node.level() * config_.connectivity + config_.connectivity_base;
            std::size_t edges = 0;
            for (level_t level = 0; level <= node.level(); ++level)
                edges += neighbors_(node, level).size();

            result.allocated_bytes += node_bytes_(node).size();
            result.edges += edges;
            result.max_edges += max_edges;
        }
        return result;
    }

    stats_t stats(std::size_t level) const noexcept {
        stats_t result{};
        result.nodes = size();

        std::size_t neighbors_bytes = !level ? pre_.neighbors_base_bytes : pre_.neighbors_bytes;
        for (std::size_t i = 0; i != result.nodes; ++i) {
            node_t node = node_at_(i);
            if (static_cast<std::size_t>(node.level()) < level)
                continue;

            result.edges += neighbors_(node, level).size();
            result.allocated_bytes += node_head_bytes_() + neighbors_bytes;
        }

        std::size_t max_edges_per_node = level ? config_.connectivity_base : config_.connectivity;
        result.max_edges = result.nodes * max_edges_per_node;
        return result;
    }

    /**
     *  @brief  A relatively accurate lower bound on the amount of memory consumed by the system.
     *          In practice it's error will be below 10%.
     */
    std::size_t memory_usage(std::size_t allocator_entry_bytes = default_allocator_entry_bytes()) const noexcept {
        std::size_t total = 0;
        if (!viewed_file_) {
            stats_t s = stats();
            total += s.allocated_bytes;
            total += s.nodes * allocator_entry_bytes;
        }

        // Temporary data-structures, proportional to the number of nodes:
        total += limits_.members * sizeof(node_t) + allocator_entry_bytes;

        // Temporary data-structures, proportional to the number of threads:
        total += limits_.threads() * sizeof(context_t) + allocator_entry_bytes * 3;
        return total;
    }

    std::size_t memory_usage_per_node(level_t level) const noexcept { return node_bytes_(level); }

    void change_metric(metric_t const& m) noexcept {
        metric_ = m;
        for (std::size_t i = 0; i != limits_.threads(); ++i)
            contexts_[i].metric = metric_;
    }

#pragma endregion

#pragma region Serialization

    /**
     *  @brief  Saves serialized binary index representation to disk,
     *          co-locating vectors and neighbors lists.
     *          Available on Linux, MacOS, Windows.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t save(output_file_t file, progress_at&& progress = {}) const noexcept {

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        // Export some basic metadata
        index_serialized_header_t header;
        header.size = size_;
        header.connectivity = config_.connectivity;
        header.connectivity_base = config_.connectivity_base;
        header.max_level = max_level_;
        header.entry_idx = entry_idx_;
        result = file.write(&header, sizeof(header));
        if (!result)
            return result;

        // Export the number of levels per node
        // That is both enough to estimate the overall memory consumption,
        // and to be able to estimate the offsets of every entry in the file.
        for (std::size_t i = 0; i != header.size; ++i) {
            node_t node = node_at_(i);
            level_t level = node.level();
            result = file.write(&level, sizeof(level));
            if (!result)
                return result;
        }

        // After that dump the nodes themselves
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = node_bytes_(node_at_(i));
            result = file.write(node_bytes.data(), node_bytes.size());
            if (!result)
                return result;
            progress(i, header.size);
        }

        return {};
    }

    /**
     *  @brief  Loads the serialized binary index representation from disk,
     *          copying both vectors and neighbors lists into RAM.
     *          Available on Linux, MacOS, Windows.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t load(input_file_t file, progress_at&& progress = {}) noexcept {

        // Remove previously stored objects
        reset();

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        // Pull basic metadata
        index_serialized_header_t header;
        result = file.read(&header, sizeof(header));
        if (!result)
            return result;

        // Allocate some dynamic memory to read all the levels
        level_t* levels = (level_t*)dynamic_allocator_.allocate(header.size * sizeof(level_t));
        if (!levels)
            return result.failed("Out of memory");
        result = file.read(levels, header.size * sizeof(level_t));
        if (!result)
            return result;

        // Submit metadata
        config_.connectivity = header.connectivity;
        config_.connectivity_base = header.connectivity_base;
        pre_ = precompute_(config_);
        index_limits_t limits;
        limits.members = header.size;
        if (!reserve(limits)) {
            reset();
            return result.failed("Out of memory");
        }
        size_ = header.size;
        max_level_ = static_cast<level_t>(header.max_level);
        entry_idx_ = static_cast<internal_id_t>(header.entry_idx);

        // Load the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = node_malloc_(levels[i]);
            result = file.read(node_bytes.data(), node_bytes.size());
            if (!result) {
                reset();
                return result;
            }
            nodes_[i] = node_t{node_bytes.data()};
            progress(i, header.size);
        }
        return {};
    }

    /**
     *  @brief  Memory-maps the serialized binary index representation from disk,
     *          @b without copying the vectors and neighbors lists into RAM.
     *          Available on Linux, MacOS, but @b not on Windows.
     *
     *  @param[in] file_path File where the index is saved.
     *  @param[in] executor Thread-pool to execute the job in parallel.
     *  @param[in] progress Callback to report the execution progress.
     */
    template <                                   //
        typename executor_at = dummy_executor_t, //
        typename progress_at = dummy_progress_t  //
        >
    serialization_result_t view(   //
        memory_mapped_file_t file, //
        std::size_t offset = 0,    //
        progress_at&& progress = {}) noexcept {

        // Remove previously stored objects
        reset();

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        // Pull basic metadata
        index_serialized_header_t header;
        if (file.size() - offset < sizeof(header))
            return result.failed("File is corrupted and lacks a header");
        std::memcpy(&header, file.data() + offset, sizeof(header));

        // Precompute offsets of every node
        // This could have been done with `std::exclusive_scan`, but it's only available from C++17.
        std::size_t* offsets = (std::size_t*)dynamic_allocator_.allocate(header.size * sizeof(std::size_t));
        misaligned_ptr_gt<level_t> levels{(byte_t*)file.data() + offset + sizeof(header)};
        if (!offsets)
            return result.failed("Out of memory");
        offsets[0] = offset + sizeof(header) + sizeof(level_t) * header.size;
        for (std::size_t i = 1; i < header.size; ++i)
            offsets[i] = offsets[i - 1] + node_bytes_(levels[i - 1]);

        std::size_t total_bytes = offsets[header.size - 1] + node_bytes_(levels[header.size - 1]);
        if (file.size() - offset < total_bytes)
            return result.failed("File is corrupted and can't fit all the nodes");

        // Submit metadata
        config_.connectivity = header.connectivity;
        config_.connectivity_base = header.connectivity_base;
        pre_ = precompute_(config_);
        index_limits_t limits;
        limits.members = header.size;
        if (!reserve(limits)) {
            reset();
            return result.failed("Out of memory");
        }
        size_ = header.size;
        max_level_ = static_cast<level_t>(header.max_level);
        entry_idx_ = static_cast<internal_id_t>(header.entry_idx);

        // Rapidly address all the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            nodes_[i] = node_t{(byte_t*)file.data() + offsets[i]};
            progress(i, header.size);
        }
        viewed_file_ = std::move(file);
        return {};
    }

#pragma endregion

    /**
     *  @brief  Scans the whole collection, removing the links leading towards
     *          banned entries. This essentially isolates some nodes from the rest
     *          of the graph, while keeping their outgoing links, in case the node
     *          is structurally relevant and has a crucial role in the index.
     *          It won't reclaim the memory.
     *
     *  @param[in] allow_member Predicate to mark nodes for isolation.
     *  @param[in] executor Thread-pool to execute the job in parallel.
     *  @param[in] progress Callback to report the execution progress.
     */
    template <                                        //
        typename allow_member_at = dummy_predicate_t, //
        typename executor_at = dummy_executor_t,      //
        typename progress_at = dummy_progress_t       //
        >
    void isolate(                               //
        allow_member_at&& allow_member,         //
        executor_at&& executor = executor_at{}, //
        progress_at&& progress = progress_at{}) noexcept {

        // Erase all the incoming links
        executor.execute_bulk(size(), [&](std::size_t, std::size_t node_idx) {
            node_t node = node_at_(node_idx);
            for (level_t level = 0; level <= node.level(); ++level) {
                neighbors_ref_t neighbors = neighbors_(node, level);
                std::size_t old_size = neighbors.size();
                neighbors.clear();
                for (std::size_t i = 0; i != old_size; ++i) {
                    internal_id_t neighbor_id = neighbors[i];
                    node_t neighbor = node_at_(neighbor_id);
                    if (allow_member(member_cref_t{neighbor.label(), neighbor_id}))
                        neighbors.push_back(neighbor_id);
                }
            }
        });
    }

  private:
    inline static precomputed_constants_t precompute_(index_config_t const& config) noexcept {
        precomputed_constants_t pre;
        pre.inverse_log_connectivity = 1.0 / std::log(static_cast<double>(config.connectivity));
        pre.neighbors_bytes = config.connectivity * sizeof(internal_id_t) + sizeof(neighbors_count_t);
        pre.neighbors_base_bytes = config.connectivity_base * sizeof(internal_id_t) + sizeof(neighbors_count_t);
        return pre;
    }

    using span_bytes_t = span_gt<byte_t>;

    inline span_bytes_t node_bytes_(node_t node) const noexcept { return {node.tape(), node_bytes_(node.level())}; }
    inline std::size_t node_bytes_(level_t level) const noexcept {
        return node_head_bytes_() + node_neighbors_bytes_(level);
    }
    inline std::size_t node_neighbors_bytes_(node_t node) const noexcept { return node_neighbors_bytes_(node.level()); }
    inline std::size_t node_neighbors_bytes_(level_t level) const noexcept {
        return pre_.neighbors_base_bytes + pre_.neighbors_bytes * level;
    }

    span_bytes_t node_malloc_(level_t level) noexcept {
        std::size_t node_bytes = node_bytes_(level);
        byte_t* data = (byte_t*)tape_allocator_.allocate(node_bytes);
        return data ? span_bytes_t{data, node_bytes} : span_bytes_t{};
    }

    node_t node_make_(label_t label, level_t level) noexcept {
        span_bytes_t node_bytes = node_malloc_(level);
        if (!node_bytes)
            return {};

        std::memset(node_bytes.data(), 0, node_bytes.size());
        node_t node{(byte_t*)node_bytes.data()};
        node.label(label);
        node.level(level);
        return node;
    }

    node_t node_make_copy_(span_bytes_t old_bytes) noexcept {
        byte_t* data = (byte_t*)tape_allocator_.allocate(old_bytes.size());
        if (!data)
            return {};
        std::memcpy(data, old_bytes.data(), old_bytes.size());
        return node_t{data};
    }

    void node_free_(std::size_t idx) noexcept {
        if (viewed_file_)
            return;

        node_t& node = nodes_[idx];
        tape_allocator_.deallocate(node.tape(), node_bytes_(node).size());
        node = node_t{};
    }

    inline node_t node_at_(std::size_t idx) const noexcept { return nodes_[idx]; }
    inline neighbors_ref_t neighbors_base_(node_t node) const noexcept { return {node.neighbors_tape()}; }

    inline neighbors_ref_t neighbors_non_base_(node_t node, level_t level) const noexcept {
        return {node.neighbors_tape() + pre_.neighbors_base_bytes + (level - 1) * pre_.neighbors_bytes};
    }

    inline neighbors_ref_t neighbors_(node_t node, level_t level) const noexcept {
        return level ? neighbors_non_base_(node, level) : neighbors_base_(node);
    }

    struct node_lock_t {
        visits_bitset_t& bitset;
        std::size_t idx;

        inline ~node_lock_t() noexcept { bitset.atomic_reset(idx); }
    };

    inline node_lock_t node_lock_(std::size_t idx) const noexcept {
        while (nodes_mutexes_.atomic_set(idx))
            ;
        return {nodes_mutexes_, idx};
    }

    template <typename content_at>
    void connect_node_across_levels_(                                     //
        internal_id_t node_id, content_at&& content,                      //
        internal_id_t entry_idx, level_t max_level, level_t target_level, //
        add_config_t const& config, context_t& context) usearch_noexcept_m {

        // Go down the level, tracking only the closest match
        internal_id_t closest_id = search_for_one_(entry_idx, content, max_level, target_level, context);

        // From `target_level` down perform proper extensive search
        for (level_t level = (std::min)(target_level, max_level); level >= 0; --level) {
            // TODO: Handle out of memory conditions
            search_to_insert_(closest_id, content, level, config.expansion, context);
            closest_id = connect_new_node_(node_id, level, context);
            reconnect_neighbor_nodes_(node_id, level, context);
        }
    }

    internal_id_t connect_new_node_(internal_id_t new_id, level_t level, context_t& context) usearch_noexcept_m {

        node_t new_node = node_at_(new_id);
        top_candidates_t& top = context.top_candidates;

        // Outgoing links from `new_id`:
        neighbors_ref_t new_neighbors = neighbors_(new_node, level);
        {
            usearch_assert_m(!new_neighbors.size(), "The newly inserted element should have blank link list");
            candidates_view_t top_view = refine_(top, config_.connectivity, context);

            for (std::size_t idx = 0; idx != top_view.size(); idx++) {
                usearch_assert_m(!new_neighbors[idx], "Possible memory corruption");
                usearch_assert_m(level <= node_at_(top_view[idx].id).level(), "Linking to missing level");
                new_neighbors.push_back(top_view[idx].id);
            }
        }

        return new_neighbors[0];
    }

    void reconnect_neighbor_nodes_(internal_id_t new_id, level_t level, context_t& context) usearch_noexcept_m {

        node_t new_node = node_at_(new_id);
        top_candidates_t& top = context.top_candidates;
        neighbors_ref_t new_neighbors = neighbors_(new_node, level);

        // Reverse links from the neighbors:
        std::size_t const connectivity_max = level ? config_.connectivity : config_.connectivity_base;
        for (internal_id_t close_id : new_neighbors) {
            node_t close_node = node_at_(close_id);
            node_lock_t close_lock = node_lock_(close_id);

            neighbors_ref_t close_header = neighbors_(close_node, level);
            usearch_assert_m(close_header.size() <= connectivity_max, "Possible corruption");
            usearch_assert_m(close_id != new_id, "Self-loops are impossible");
            usearch_assert_m(level <= close_node.level(), "Linking to missing level");

            // If `new_id` is already present in the neighboring connections of `close_id`
            // then no need to modify any connections or run the heuristics.
            if (close_header.size() < connectivity_max) {
                close_header.push_back(new_id);
                continue;
            }

            // To fit a new connection we need to drop an existing one.
            top.clear();
            usearch_assert_m((top.reserve(close_header.size() + 1)), "The memory must have been reserved in `add`");
            top.insert_reserved({context.measure(new_node, close_node), new_id});
            for (internal_id_t successor_id : close_header)
                top.insert_reserved({context.measure(node_at_(successor_id), close_node), successor_id});

            // Export the results:
            close_header.clear();
            candidates_view_t top_view = refine_(top, connectivity_max, context);
            for (std::size_t idx = 0; idx != top_view.size(); idx++)
                close_header.push_back(top_view[idx].id);
        }
    }

    level_t choose_random_level_(std::default_random_engine& level_generator) const noexcept {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -std::log(distribution(level_generator)) * pre_.inverse_log_connectivity;
        return (level_t)r;
    }

    template <typename query_at>
    internal_id_t search_for_one_(                  //
        internal_id_t closest_id, query_at&& query, //
        level_t begin_level, level_t end_level,     //
        context_t& context) const noexcept {

        distance_t closest_dist = context.measure(query, node_at_(closest_id));
        for (level_t level = begin_level; level > end_level; --level) {
            bool changed;
            do {
                changed = false;
                node_t closest_node = node_at_(closest_id);
                node_lock_t closest_lock = node_lock_(closest_id);
                neighbors_ref_t closest_neighbors = neighbors_non_base_(closest_node, level);
                for (internal_id_t candidate_id : closest_neighbors) {
                    distance_t candidate_dist = context.measure(query, node_at_(candidate_id));
                    if (candidate_dist < closest_dist) {
                        closest_dist = candidate_dist;
                        closest_id = candidate_id;
                        changed = true;
                    }
                }
                context.iteration_cycles++;
            } while (changed);
        }
        return closest_id;
    }

    /**
     *  @brief  Traverses a layer of a graph, to find the best place to insert a new node.
     *          Locks the nodes in the process, assuming other threads are updating neighbors lists.
     *  @return `true` if procedure succeeded, `false` if run out of memory.
     */
    template <typename query_at>
    bool search_to_insert_( //
        internal_id_t start_id, query_at&& query, level_t level, std::size_t top_limit, context_t& context) noexcept {

        visits_bitset_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates; // pop min, push
        top_candidates_t& top = context.top_candidates;    // pop max, push

        visits.clear();
        next.clear();
        top.clear();

        distance_t radius = context.measure(query, node_at_(start_id));
        next.insert_reserved({-radius, start_id});
        top.insert_reserved({radius, start_id});
        visits.set(start_id);

        while (!next.empty()) {

            candidate_t candidacy = next.top();
            if ((-candidacy.distance) > radius && top.size() == top_limit)
                break;

            next.pop();
            context.iteration_cycles++;

            internal_id_t candidate_id = candidacy.id;
            node_t candidate_ref = node_at_(candidate_id);
            node_lock_t candidate_lock = node_lock_(candidate_id);
            neighbors_ref_t candidate_neighbors = neighbors_(candidate_ref, level);

            prefetch_neighbors_(candidate_neighbors, visits);
            for (internal_id_t successor_id : candidate_neighbors) {
                if (visits.test(successor_id))
                    continue;

                visits.set(successor_id);
                distance_t successor_dist = context.measure(query, node_at_(successor_id));

                if (top.size() < top_limit || successor_dist < radius) {
                    // This can substantially grow our priority queue:
                    next.insert({-successor_dist, successor_id});
                    // This will automatically evict poor matches:
                    top.insert({successor_dist, successor_id}, top_limit);
                    radius = top.top().distance;
                }
            }
        }
        return true;
    }

    /**
     *  @brief  Traverses the @b base layer of a graph, to find a close match.
     *          Doesn't lock any nodes, assuming read-only simultaneous access.
     *  @return `true` if procedure succeeded, `false` if run out of memory.
     */
    template <typename query_at, typename predicate_at>
    bool search_to_find_in_base_( //
        internal_id_t start_id, query_at&& query, std::size_t expansion, context_t& context,
        predicate_at&& predicate) const noexcept {

        visits_bitset_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates; // pop min, push
        top_candidates_t& top = context.top_candidates;    // pop max, push
        std::size_t const top_limit = expansion;

        visits.clear();
        next.clear();
        top.clear();

        distance_t radius = context.measure(query, node_at_(start_id));
        next.insert_reserved({-radius, start_id});
        top.insert_reserved({radius, start_id});
        visits.set(start_id);

        while (!next.empty()) {

            candidate_t candidate = next.top();
            if ((-candidate.distance) > radius)
                break;

            next.pop();
            context.iteration_cycles++;

            internal_id_t candidate_id = candidate.id;
            neighbors_ref_t candidate_neighbors = neighbors_base_(node_at_(candidate_id));

            prefetch_neighbors_(candidate_neighbors, visits);
            for (internal_id_t successor_id : candidate_neighbors) {
                if (visits.test(successor_id))
                    continue;

                visits.set(successor_id);
                node_t successor = node_at_(successor_id);
                distance_t successor_dist = context.measure(query, successor);

                if (top.size() < top_limit || successor_dist < radius) {
                    // This can substantially grow our priority queue:
                    next.insert({-successor_dist, successor_id});
                    if (predicate( //
                            match_t{member_cref_t{successor.label(), successor_id}, successor_dist})) {
                        // This will automatically evict poor matches:
                        top.insert({successor_dist, successor_id}, top_limit);
                        radius = top.top().distance;
                    }
                }
            }
        }

        return true;
    }

    /**
     *  @brief  Iterates through all members, without actually touching the index.
     */
    template <typename query_at, typename predicate_at>
    void search_exact_(                                          //
        query_at&& query, std::size_t count, context_t& context, //
        predicate_at&& predicate) const noexcept {

        top_candidates_t& top = context.top_candidates;
        top.clear();
        top.reserve(count);
        for (std::size_t i = 0; i != size(); ++i) {
            internal_id_t id = static_cast<internal_id_t>(i);
            node_t node = node_at_(i);
            distance_t distance = context.measure(query, node);
            if (predicate(match_t{member_cref_t{node.label(), id}, distance}))
                top.insert(candidate_t{distance, id}, count);
        }
    }

    void prefetch_neighbors_(neighbors_ref_t, visits_bitset_t const&) const noexcept {}

    /**
     *  @brief  This algorithm from the original paper implements a heuristic,
     *          that massively reduces the number of connections a point has,
     *          to keep only the neighbors, that are from each other.
     */
    candidates_view_t refine_(top_candidates_t& top, std::size_t needed, context_t& context) const noexcept {

        top.sort_ascending();
        candidate_t* top_data = top.data();
        std::size_t const top_count = top.size();
        if (top_count < needed)
            return {top_data, top_count};

        std::size_t submitted_count = 1;
        std::size_t consumed_count = 1; /// Always equal or greater than `submitted_count`.
        while (submitted_count < needed && consumed_count < top_count) {
            candidate_t candidate = top_data[consumed_count];
            node_t candidate_ref = node_at_(candidate.id);
            distance_t candidate_dist = candidate.distance;
            bool good = true;
            for (std::size_t idx = 0; idx < submitted_count; idx++) {
                candidate_t submitted = top_data[idx];
                node_t submitted_node = node_at_(submitted.id);
                distance_t inter_result_dist = context.measure(submitted_node, candidate_ref);
                if (inter_result_dist < candidate_dist) {
                    good = false;
                    break;
                }
            }

            if (good) {
                top_data[submitted_count] = top_data[consumed_count];
                submitted_count++;
            }
            consumed_count++;
        }

        top.shrink(submitted_count);
        return {top_data, submitted_count};
    }
};

#if 0

/**
 *  @brief  Extracts metadata from pre-constructed index on disk, without loading it
 *          or mapping the whole binary file.
 */
inline file_head_result_t index_metadata(char const* file_path) noexcept {

    file_head_result_t result;
    file_header_t state_buffer{};
    std::FILE* file = std::fopen(file_path, "rb");
    if (!file)
        return result.failed(std::strerror(errno));

    // Read the header
    std::size_t read = std::fread(&state_buffer[0], sizeof(file_header_t), 1, file);
    std::fclose(file);
    if (!read)
        return result.failed(std::strerror(errno));

    // Parse and validate the MIME type
    file_head_t state{state_buffer};
    if (std::strncmp(state.magic, default_magic(), std::strlen(default_magic())) != 0)
        return result.failed("Wrong MIME type!");

    result.version_major = state.version_major;
    result.version_minor = state.version_minor;
    result.version_patch = state.version_patch;
    result.connectivity = state.connectivity;
    result.max_level = state.max_level;
    result.bytes_per_label = state.bytes_per_label;
    result.bytes_per_id = state.bytes_per_id;
    result.size = state.size;
    result.entry_idx = state.entry_idx;
    result.bytes_total = state.bytes_total;
    result.bytes_for_vectors = state.bytes_for_vectors;
    result.bytes_checksum = state.bytes_checksum;
    return result;
}


    struct join_result_t {
        error_t error{};
        std::size_t intersection_size{};
        std::size_t engagements{};
        std::size_t cycles{};
        std::size_t measurements{};

        explicit operator bool() const noexcept { return !error; }
        join_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

/**
 *  @brief  Adapts the Male-Optimal Stable Marriage algorithm for unequal sets
 *          to perform fast one-to-one matching between two large collections
 *          of vectors, using approximate nearest neighbors search.
 *
 *  @param[inout] first_to_second Container to map ::first labels to ::second.
 *  @param[inout] second_to_first Container to map ::second labels to ::first.
 *  @param[in] executor Thread-pool to execute the job in parallel.
 *  @param[in] progress Callback to report the execution progress.
 */
template <                                                        //
    typename first_to_second_at = dummy_label_to_label_mapping_t, //
    typename second_to_first_at = dummy_label_to_label_mapping_t, //
    typename executor_at = dummy_executor_t,                      //
    typename progress_at = dummy_progress_t                       //
    >
static join_result_t join(                                       //
    index_gt const& first, index_gt const& second,               //
    join_config_t config = {},                                   //
    first_to_second_at&& first_to_second = first_to_second_at{}, //
    second_to_first_at&& second_to_first = second_to_first_at{}, //
    executor_at&& executor = executor_at{},                      //
    progress_at&& progress = progress_at{}) noexcept {

    if (second.size() < first.size())
        return join_small_and_big_(                            //
            second, first,                                     //
            config,                                            //
            std::forward<second_to_first_at>(second_to_first), //
            std::forward<first_to_second_at>(first_to_second), //
            std::forward<executor_at>(executor),               //
            std::forward<progress_at>(progress));
    else
        return join_small_and_big_(                            //
            first, second,                                     //
            config,                                            //
            std::forward<first_to_second_at>(first_to_second), //
            std::forward<second_to_first_at>(second_to_first), //
            std::forward<executor_at>(executor),               //
            std::forward<progress_at>(progress));
}

template <typename first_to_second_at, typename second_to_first_at, typename executor_at, typename progress_at>
static join_result_t join_small_and_big_(       //
    index_gt const& men, index_gt const& women, //
    join_config_t config,                       //
    first_to_second_at&& man_to_woman,          //
    second_to_first_at&& woman_to_man,          //
    executor_at&& executor,                     //
    progress_at&& progress) noexcept {

    join_result_t result;

    // Sanity checks and argument validation:
    if (&men == &women)
        return result.failed("Can't join with itself, consider copying");
    if (config.max_proposals == 0)
        config.max_proposals = std::log(men.size()) + executor.size();
    config.max_proposals = (std::min)(men.size(), config.max_proposals);

    using proposals_count_t = std::uint16_t;
    using ids_allocator_t = typename allocator_traits_t::template rebind_alloc<internal_id_t>;
    dynamic_allocator_t& alloc = men.dynamic_allocator_;

    // Create an atomic queue, as a ring structure, from/to which
    // free men will be added/pulled.
    std::mutex free_men_mutex{};
    ring_gt<internal_id_t, ids_allocator_t> free_men;
    free_men.reserve(men.size());
    for (std::size_t i = 0; i != men.size(); ++i)
        free_men.push(static_cast<internal_id_t>(i));

    // We are gonna need some temporary memory.
    proposals_count_t* proposal_counts = (proposals_count_t*)alloc.allocate(sizeof(proposals_count_t) * men.size());
    internal_id_t* man_to_woman_ids = (internal_id_t*)alloc.allocate(sizeof(internal_id_t) * men.size());
    internal_id_t* woman_to_man_ids = (internal_id_t*)alloc.allocate(sizeof(internal_id_t) * women.size());
    if (!proposal_counts || !man_to_woman_ids || !woman_to_man_ids)
        return result.failed("Can't temporary mappings");

    internal_id_t missing_id;
    std::memset((void*)&missing_id, 0xFF, sizeof(internal_id_t));
    std::memset((void*)man_to_woman_ids, 0xFF, sizeof(internal_id_t) * men.size());
    std::memset((void*)woman_to_man_ids, 0xFF, sizeof(internal_id_t) * women.size());
    std::memset(proposal_counts, 0, sizeof(proposals_count_t) * men.size());

    // Define locks, to limit concurrent accesses to `man_to_woman_ids` and `woman_to_man_ids`.
    visits_bitset_t men_locks, women_locks;
    if (!men_locks.resize(men.size()) || !women_locks.resize(women.size()))
        return result.failed("Can't allocate locks");

    // Accumulate statistics from all the contexts,
    // to have a baseline to compare with, by the time the `join` is finished.
    std::size_t old_measurements{};
    std::size_t old_cycles{};
    for (std::size_t thread_idx = 0; thread_idx != executor.size(); ++thread_idx) {
        old_measurements += women.contexts_[thread_idx].measurements_count;
        old_cycles += women.contexts_[thread_idx].iteration_cycles;
    }
    std::atomic<std::size_t> rounds{0};
    std::atomic<std::size_t> engagements{0};

    // Concurrently process all the men
    executor.execute_bulk([&](std::size_t thread_idx) {
        context_t& context = women.contexts_[thread_idx];
        search_config_t search_config;
        search_config.expansion = config.expansion;
        search_config.exact = config.exact;
        search_config.thread = thread_idx;
        internal_id_t free_man_id;

        // While there exist a free man who still has a woman to propose to.
        while (true) {
            std::size_t passed_rounds = 0;
            std::size_t total_rounds = 0;
            {
                std::unique_lock<std::mutex> pop_lock(free_men_mutex);
                if (!free_men.try_pop(free_man_id))
                    // Primary exit path, we have exhausted the list of candidates
                    break;
                passed_rounds = ++rounds;
                total_rounds = passed_rounds + free_men.size();
            }
            progress(passed_rounds, total_rounds);
            while (men_locks.atomic_set(free_man_id))
                ;

            node_t free_man = men.node_at_(free_man_id);
            proposals_count_t& free_man_proposals = proposal_counts[free_man_id];
            if (free_man_proposals >= config.max_proposals)
                continue;

            // Find the closest woman, to whom this man hasn't proposed yet.
            free_man_proposals++;
            search_result_t candidates = women.search(free_man.vector_view(), free_man_proposals, search_config);
            if (!candidates) {
                // TODO:
            }

            match_t match = candidates.back();
            member_cref_t woman = match.member;
            while (women_locks.atomic_set(woman.id))
                ;

            internal_id_t husband_id = woman_to_man_ids[woman.id];
            bool woman_is_free = husband_id == missing_id;
            if (woman_is_free) {
                // Engagement
                man_to_woman_ids[free_man_id] = woman.id;
                woman_to_man_ids[woman.id] = free_man_id;
                engagements++;
            } else {
                distance_t distance_from_husband = context.measure(woman.vector, men.node_at_(husband_id));
                distance_t distance_from_candidate = match.distance;
                if (distance_from_husband > distance_from_candidate) {
                    // Break-up
                    while (men_locks.atomic_set(husband_id))
                        ;
                    man_to_woman_ids[husband_id] = missing_id;
                    men_locks.atomic_reset(husband_id);

                    // New Engagement
                    man_to_woman_ids[free_man_id] = woman.id;
                    woman_to_man_ids[woman.id] = free_man_id;
                    engagements++;

                    std::unique_lock<std::mutex> push_lock(free_men_mutex);
                    free_men.push(husband_id);
                } else {
                    std::unique_lock<std::mutex> push_lock(free_men_mutex);
                    free_men.push(free_man_id);
                }
            }

            men_locks.atomic_reset(free_man_id);
            women_locks.atomic_reset(woman.id);
        }
    });

    // Export the IDs into labels:
    std::size_t intersection_size = 0;
    for (std::size_t i = 0; i != men.size(); ++i) {
        internal_id_t woman_id = man_to_woman_ids[i];
        if (woman_id != missing_id) {
            label_t man = men.node_at_(i).label();
            label_t woman = women.node_at_(woman_id).label();
            man_to_woman[man] = woman;
            woman_to_man[woman] = man;
            intersection_size++;
        }
    }

    // Deallocate memory
    alloc.deallocate((byte_t*)proposal_counts, sizeof(proposals_count_t) * men.size());
    alloc.deallocate((byte_t*)man_to_woman_ids, sizeof(internal_id_t) * men.size());
    alloc.deallocate((byte_t*)woman_to_man_ids, sizeof(internal_id_t) * women.size());

    // Export stats
    result.engagements = engagements;
    result.intersection_size = intersection_size;
    for (std::size_t thread_idx = 0; thread_idx != executor.size(); ++thread_idx) {
        result.measurements += women.contexts_[thread_idx].measurements_count;
        result.cycles += women.contexts_[thread_idx].iteration_cycles;
    }
    result.measurements -= old_measurements;
    result.cycles -= old_cycles;
    return result;
}
#endif

} // namespace usearch
} // namespace unum

#endif
