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

#define USEARCH_VERSION_MAJOR 2
#define USEARCH_VERSION_MINOR 8
#define USEARCH_VERSION_PATCH 14

// Inferring C++ version
// https://stackoverflow.com/a/61552074
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define USEARCH_DEFINED_CPP17
#endif

// Inferring target OS: Windows, MacOS, or Linux
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define USEARCH_DEFINED_WINDOWS
#elif defined(__APPLE__) && defined(__MACH__)
#define USEARCH_DEFINED_APPLE
#elif defined(__linux__)
#define USEARCH_DEFINED_LINUX
#endif

// Inferring the compiler: Clang vs GCC
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
// https://stackoverflow.com/a/5273354
#if INTPTR_MAX == INT64_MAX
#define USEARCH_64BIT_ENV
#elif INTPTR_MAX == INT32_MAX
#define USEARCH_32BIT_ENV
#else
#error Unknown pointer size or missing size macros!
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
#include <cassert>
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

/// @brief  The `std::destroy_at` alternative for C++11.
template <typename at, typename sfinae_at = at>
typename std::enable_if<std::is_pod<sfinae_at>::value>::type destroy_at(at*) {}
template <typename at, typename sfinae_at = at>
typename std::enable_if<!std::is_pod<sfinae_at>::value>::type destroy_at(at* obj) {
    obj->~sfinae_at();
}

/// @brief  The `std::construct_at` alternative for C++11.
template <typename at, typename sfinae_at = at>
typename std::enable_if<std::is_pod<sfinae_at>::value>::type construct_at(at*) {}
template <typename at, typename sfinae_at = at>
typename std::enable_if<!std::is_pod<sfinae_at>::value>::type construct_at(at* obj) {
    new (obj) at();
}

/**
 *  @brief  A reference to a misaligned memory location with a specific type.
 *          It is needed to avoid Undefined Behavior when dereferencing addresses
 *          indivisible by `sizeof(at)`.
 */
template <typename at> class misaligned_ref_gt {
    using element_t = at;
    using mutable_t = typename std::remove_const<element_t>::type;
    byte_t* ptr_;

  public:
    misaligned_ref_gt(byte_t* ptr) noexcept : ptr_(ptr) {}
    operator mutable_t() const noexcept { return misaligned_load<mutable_t>(ptr_); }
    misaligned_ref_gt& operator=(mutable_t const& v) noexcept {
        misaligned_store<mutable_t>(ptr_, v);
        return *this;
    }

    void reset(byte_t* ptr) noexcept { ptr_ = ptr; }
    byte_t* ptr() const noexcept { return ptr_; }
};

/**
 *  @brief  A pointer to a misaligned memory location with a specific type.
 *          It is needed to avoid Undefined Behavior when dereferencing addresses
 *          indivisible by `sizeof(at)`.
 */
template <typename at> class misaligned_ptr_gt {
    using element_t = at;
    using mutable_t = typename std::remove_const<element_t>::type;
    byte_t* ptr_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = element_t;
    using difference_type = std::ptrdiff_t;
    using pointer = misaligned_ptr_gt<element_t>;
    using reference = misaligned_ref_gt<element_t>;

    reference operator*() const noexcept { return {ptr_}; }
    reference operator[](std::size_t i) noexcept { return reference(ptr_ + i * sizeof(element_t)); }
    value_type operator[](std::size_t i) const noexcept {
        return misaligned_load<element_t>(ptr_ + i * sizeof(element_t));
    }

    misaligned_ptr_gt(byte_t* ptr) noexcept : ptr_(ptr) {}
    misaligned_ptr_gt operator++(int) noexcept { return misaligned_ptr_gt(ptr_ + sizeof(element_t)); }
    misaligned_ptr_gt operator--(int) noexcept { return misaligned_ptr_gt(ptr_ - sizeof(element_t)); }
    misaligned_ptr_gt operator+(difference_type d) noexcept { return misaligned_ptr_gt(ptr_ + d * sizeof(element_t)); }
    misaligned_ptr_gt operator-(difference_type d) noexcept { return misaligned_ptr_gt(ptr_ - d * sizeof(element_t)); }

    // clang-format off
    misaligned_ptr_gt& operator++() noexcept { ptr_ += sizeof(element_t); return *this; }
    misaligned_ptr_gt& operator--() noexcept { ptr_ -= sizeof(element_t); return *this; }
    misaligned_ptr_gt& operator+=(difference_type d) noexcept { ptr_ += d * sizeof(element_t); return *this; }
    misaligned_ptr_gt& operator-=(difference_type d) noexcept { ptr_ -= d * sizeof(element_t); return *this; }
    // clang-format on

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
 *  @brief  Similar to `std::vector`, but doesn't support dynamic resizing.
 *          On the bright side, this can't throw exceptions.
 */
template <typename scalar_at, typename allocator_at = std::allocator<scalar_at>> class buffer_gt {
    scalar_at* data_;
    std::size_t size_;

  public:
    buffer_gt() noexcept : data_(nullptr), size_(0u) {}
    buffer_gt(std::size_t size) noexcept : data_(allocator_at{}.allocate(size)), size_(data_ ? size : 0u) {
        if (!std::is_trivially_default_constructible<scalar_at>::value)
            for (std::size_t i = 0; i != size_; ++i)
                construct_at(data_ + i);
    }
    ~buffer_gt() noexcept {
        if (!std::is_trivially_destructible<scalar_at>::value)
            for (std::size_t i = 0; i != size_; ++i)
                destroy_at(data_ + i);
        allocator_at{}.deallocate(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
    scalar_at* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    scalar_at* begin() const noexcept { return data_; }
    scalar_at* end() const noexcept { return data_ + size_; }
    operator scalar_at*() const noexcept { return data(); }
    scalar_at& operator[](std::size_t i) noexcept { return data_[i]; }
    scalar_at const& operator[](std::size_t i) const noexcept { return data_[i]; }
    explicit operator bool() const noexcept { return data_; }
    scalar_at* release() noexcept {
        size_ = 0;
        return exchange(data_, nullptr);
    }

    buffer_gt(buffer_gt const&) = delete;
    buffer_gt& operator=(buffer_gt const&) = delete;

    buffer_gt(buffer_gt&& other) noexcept : data_(exchange(other.data_, nullptr)), size_(exchange(other.size_, 0)) {}
    buffer_gt& operator=(buffer_gt&& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        return *this;
    }
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

    error_t(error_t const&) = delete;
    error_t& operator=(error_t const&) = delete;
    error_t(error_t&& other) noexcept : message_(exchange(other.message_, nullptr)) {}
    error_t& operator=(error_t&& other) noexcept {
        std::swap(message_, other.message_);
        return *this;
    }
    explicit operator bool() const noexcept { return message_ != nullptr; }
    char const* what() const noexcept { return message_; }
    char const* release() noexcept { return exchange(message_, nullptr); }

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    ~error_t() noexcept(false) {
#if defined(USEARCH_DEFINED_CPP17)
        if (message_ && std::uncaught_exceptions() == 0)
#else
        if (message_ && std::uncaught_exception() == 0)
#endif
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
 *  @brief  Light-weight bitset implementation to sync nodes updates during graph mutations.
 *          Extends basic functionality with @b atomic operations.
 */
template <typename allocator_at = std::allocator<byte_t>> class bitset_gt {
    using allocator_t = allocator_at;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    using compressed_slot_t = unsigned long;

    static constexpr std::size_t bits_per_slot() { return sizeof(compressed_slot_t) * CHAR_BIT; }
    static constexpr compressed_slot_t bits_mask() { return sizeof(compressed_slot_t) * CHAR_BIT - 1; }
    static constexpr std::size_t slots(std::size_t bits) { return divide_round_up<bits_per_slot()>(bits); }

    compressed_slot_t* slots_{};
    /// @brief size - number of bits in the bitset
    std::size_t size_{};
    /// @brief Number of slots.
    std::size_t count_{};

  public:
    bitset_gt() noexcept {}
    ~bitset_gt() noexcept { reset(); }

    explicit operator bool() const noexcept { return slots_; }
    std::size_t size() const noexcept { return size_; }
    void clear() noexcept {
        if (slots_)
            std::memset(slots_, 0, count_ * sizeof(compressed_slot_t));
    }

    void reset() noexcept {
        if (slots_)
            allocator_t{}.deallocate((byte_t*)slots_, count_ * sizeof(compressed_slot_t));
        slots_ = nullptr;
        count_ = 0;
    }

    bitset_gt(std::size_t capacity) noexcept
        : slots_((compressed_slot_t*)allocator_t{}.allocate(slots(capacity) * sizeof(compressed_slot_t))),
          size_(slots_ ? capacity : 0u), count_(slots_ ? slots(capacity) : 0u) {
        clear();
    }

    bitset_gt(bitset_gt&& other) noexcept {
        slots_ = exchange(other.slots_, nullptr);
        count_ = exchange(other.count_, 0);
        size_ = exchange(other.size_, 0);
    }

    bitset_gt& operator=(bitset_gt&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(count_, other.count_);
        std::swap(size_, other.size_);
        return *this;
    }

    bitset_gt(bitset_gt const&) = delete;
    bitset_gt& operator=(bitset_gt const&) = delete;

    inline bool test(std::size_t i) const noexcept { return slots_[i / bits_per_slot()] & (1ul << (i & bits_mask())); }
    inline bool set(std::size_t i) noexcept {
        compressed_slot_t& slot = slots_[i / bits_per_slot()];
        compressed_slot_t mask{1ul << (i & bits_mask())};
        bool value = slot & mask;
        slot |= mask;
        return value;
    }

#if defined(USEARCH_DEFINED_WINDOWS)

    inline bool atomic_set(std::size_t i) noexcept {
        compressed_slot_t mask{1ul << (i & bits_mask())};
        return InterlockedOr((long volatile*)&slots_[i / bits_per_slot()], mask) & mask;
    }

    inline void atomic_reset(std::size_t i) noexcept {
        compressed_slot_t mask{1ul << (i & bits_mask())};
        InterlockedAnd((long volatile*)&slots_[i / bits_per_slot()], ~mask);
    }

#else

    inline bool atomic_set(std::size_t i) noexcept {
        compressed_slot_t mask{1ul << (i & bits_mask())};
        return __atomic_fetch_or(&slots_[i / bits_per_slot()], mask, __ATOMIC_ACQUIRE) & mask;
    }

    inline void atomic_reset(std::size_t i) noexcept {
        compressed_slot_t mask{1ul << (i & bits_mask())};
        __atomic_fetch_and(&slots_[i / bits_per_slot()], ~mask, __ATOMIC_RELEASE);
    }

#endif

    class lock_t {
        bitset_gt& bitset_;
        std::size_t bit_offset_;

      public:
        inline ~lock_t() noexcept { bitset_.atomic_reset(bit_offset_); }
        inline lock_t(bitset_gt& bitset, std::size_t bit_offset) noexcept : bitset_(bitset), bit_offset_(bit_offset) {
            while (bitset_.atomic_set(bit_offset_))
                ;
        }
    };

    inline lock_t lock(std::size_t i) noexcept { return {*this, i}; }
};

using bitset_t = bitset_gt<>;

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
    inline uint40_t(std::uint32_t n) noexcept { std::memcpy(&octets[1], &n, 4); }

#ifdef USEARCH_64BIT_ENV
    inline uint40_t(std::uint64_t n) noexcept { std::memcpy(octets, &n, 5); }
#endif

    uint40_t(uint40_t&&) = default;
    uint40_t(uint40_t const&) = default;
    uint40_t& operator=(uint40_t&&) = default;
    uint40_t& operator=(uint40_t const&) = default;

#if defined(USEARCH_DEFINED_CLANG) && defined(USEARCH_DEFINED_APPLE)
    inline uint40_t(std::size_t n) noexcept {
#ifdef USEARCH_64BIT_ENV
        std::memcpy(octets, &n, 5);
#else
        std::memcpy(octets, &n, 4);
#endif
    }
#endif

    inline operator std::size_t() const noexcept {
        std::size_t result = 0;
#ifdef USEARCH_64BIT_ENV
        std::memcpy(&result, octets, 5);
#else
        std::memcpy(&result, octets + 1, 4);
#endif
        return result;
    }

    inline static uint40_t max() noexcept { return uint40_t{}.broadcast(0xFF); }
    inline static uint40_t min() noexcept { return uint40_t{}.broadcast(0); }
};

#if defined(USEARCH_DEFINED_WINDOWS)
#pragma pack(pop) // Reset alignment to default
#endif

static_assert(sizeof(uint40_t) == 5, "uint40_t must be exactly 5 bytes");

// clang-format off
template <typename key_at, typename std::enable_if<std::is_integral<key_at>::value>::type* = nullptr> key_at default_free_value() { return std::numeric_limits<key_at>::max(); }
template <typename key_at, typename std::enable_if<std::is_same<key_at, uint40_t>::value>::type* = nullptr> uint40_t default_free_value() { return uint40_t::max(); }
template <typename key_at, typename std::enable_if<!std::is_integral<key_at>::value && !std::is_same<key_at, uint40_t>::value>::type* = nullptr> key_at default_free_value() { return key_at(); }
// clang-format on

template <typename element_at> struct hash_gt {
    std::size_t operator()(element_at const& element) const noexcept { return std::hash<element_at>{}(element); }
};

template <> struct hash_gt<uint40_t> {
    std::size_t operator()(uint40_t const& element) const noexcept { return std::hash<std::size_t>{}(element); }
};

/**
 *  @brief  Minimalistic hash-set implementation to track visited nodes during graph traversal.
 *
 *  It doesn't support deletion of separate objects, but supports `clear`-ing all at once.
 *  It expects `reserve` to be called ahead of all insertions, so no resizes are needed.
 *  It also assumes `0xFF...FF` slots to be unused, to simplify the design.
 *  It uses linear probing, the number of slots is always a power of two, and it uses linear-probing
 *  in case of bucket collisions.
 */
template <typename element_at, typename hasher_at = hash_gt<element_at>, typename allocator_at = std::allocator<byte_t>>
class growing_hash_set_gt {

    using element_t = element_at;
    using hasher_t = hasher_at;

    using allocator_t = allocator_at;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    element_t* slots_{};
    /// @brief Number of slots.
    std::size_t capacity_{};
    /// @brief Number of populated.
    std::size_t count_{};
    hasher_t hasher_{};

  public:
    growing_hash_set_gt() noexcept {}
    ~growing_hash_set_gt() noexcept { reset(); }

    explicit operator bool() const noexcept { return slots_; }
    std::size_t size() const noexcept { return count_; }

    void clear() noexcept {
        if (slots_)
            std::memset((void*)slots_, 0xFF, capacity_ * sizeof(element_t));
        count_ = 0;
    }

    void reset() noexcept {
        if (slots_)
            allocator_t{}.deallocate((byte_t*)slots_, capacity_ * sizeof(element_t));
        slots_ = nullptr;
        capacity_ = 0;
        count_ = 0;
    }

    growing_hash_set_gt(std::size_t capacity) noexcept
        : slots_((element_t*)allocator_t{}.allocate(ceil2(capacity) * sizeof(element_t))),
          capacity_(slots_ ? ceil2(capacity) : 0u), count_(0u) {
        clear();
    }

    growing_hash_set_gt(growing_hash_set_gt&& other) noexcept {
        slots_ = exchange(other.slots_, nullptr);
        capacity_ = exchange(other.capacity_, 0);
        count_ = exchange(other.count_, 0);
    }

    growing_hash_set_gt& operator=(growing_hash_set_gt&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(capacity_, other.capacity_);
        std::swap(count_, other.count_);
        return *this;
    }

    growing_hash_set_gt(growing_hash_set_gt const&) = delete;
    growing_hash_set_gt& operator=(growing_hash_set_gt const&) = delete;

    inline bool test(element_t const& elem) const noexcept {
        std::size_t index = hasher_(elem) & (capacity_ - 1);
        while (slots_[index] != default_free_value<element_t>()) {
            if (slots_[index] == elem)
                return true;

            index = (index + 1) & (capacity_ - 1);
        }
        return false;
    }

    /**
     *
     *  @return Similar to `bitset_gt`, returns the previous value.
     */
    inline bool set(element_t const& elem) noexcept {
        std::size_t index = hasher_(elem) & (capacity_ - 1);
        while (slots_[index] != default_free_value<element_t>()) {
            // Already exists
            if (slots_[index] == elem)
                return true;

            index = (index + 1) & (capacity_ - 1);
        }
        slots_[index] = elem;
        ++count_;
        return false;
    }

    bool reserve(std::size_t new_capacity) noexcept {
        new_capacity = (new_capacity * 5u) / 3u;
        if (new_capacity <= capacity_)
            return true;

        new_capacity = ceil2(new_capacity);
        element_t* new_slots = (element_t*)allocator_t{}.allocate(new_capacity * sizeof(element_t));
        if (!new_slots)
            return false;

        std::memset((void*)new_slots, 0xFF, new_capacity * sizeof(element_t));
        std::size_t new_count = count_;
        if (count_) {
            for (std::size_t old_index = 0; old_index != capacity_; ++old_index) {
                if (slots_[old_index] == default_free_value<element_t>())
                    continue;

                std::size_t new_index = hasher_(slots_[old_index]) & (new_capacity - 1);
                while (new_slots[new_index] != default_free_value<element_t>())
                    new_index = (new_index + 1) & (new_capacity - 1);
                new_slots[new_index] = slots_[old_index];
            }
        }

        reset();
        slots_ = new_slots;
        capacity_ = new_capacity;
        count_ = new_count;
        return true;
    }
};

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
        if (n <= capacity())
            return true;
        n = (std::max<std::size_t>)(ceil2(n), 64u);
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

    inline index_config_t() = default;
    inline index_config_t(std::size_t c) noexcept
        : connectivity(c ? c : default_connectivity()), connectivity_base(c ? c * 2 : default_connectivity() * 2) {}
    inline index_config_t(std::size_t c, std::size_t cb) noexcept
        : connectivity(c), connectivity_base((std::max)(c, cb)) {}
};

struct index_limits_t {
    std::size_t members = 0;
    std::size_t threads_add = std::thread::hardware_concurrency();
    std::size_t threads_search = std::thread::hardware_concurrency();

    inline index_limits_t(std::size_t n, std::size_t t) noexcept : members(n), threads_add(t), threads_search(t) {}
    inline index_limits_t(std::size_t n = 0) noexcept : index_limits_t(n, std::thread::hardware_concurrency()) {}
    inline std::size_t threads() const noexcept { return (std::max)(threads_add, threads_search); }
    inline std::size_t concurrency() const noexcept { return (std::min)(threads_add, threads_search); }
};

struct index_update_config_t {
    /// @brief Hyper-parameter controlling the quality of indexing.
    /// Defaults to 40 in FAISS and 200 in hnswlib.
    /// > It is called `efConstruction` in the paper.
    std::size_t expansion = default_expansion_add();

    /// @brief Optional thread identifier for multi-threaded construction.
    std::size_t thread = 0;
};

struct index_search_config_t {
    /// @brief Hyper-parameter controlling the quality of search.
    /// Defaults to 16 in FAISS and 10 in hnswlib.
    /// > It is called `ef` in the paper.
    std::size_t expansion = default_expansion_search();

    /// @brief Optional thread identifier for multi-threaded construction.
    std::size_t thread = 0;

    /// @brief Brute-forces exhaustive search over all entries in the index.
    bool exact = false;
};

struct index_cluster_config_t {
    /// @brief Hyper-parameter controlling the quality of search.
    /// Defaults to 16 in FAISS and 10 in hnswlib.
    /// > It is called `ef` in the paper.
    std::size_t expansion = default_expansion_search();

    /// @brief Optional thread identifier for multi-threaded construction.
    std::size_t thread = 0;
};

struct index_copy_config_t {};

struct index_join_config_t {
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

/**
 *  @brief  An example of what a USearch-compatible ad-hoc filter would look like.
 *
 *  A similar function object can be passed to search queries to further filter entries
 *  on their auxiliary properties, such as some categorical keys stored in an external DBMS.
 */
struct dummy_predicate_t {
    template <typename member_at> constexpr bool operator()(member_at&&) const noexcept { return true; }
};

/**
 *  @brief  An example of what a USearch-compatible ad-hoc operation on in-flight entries.
 *
 *  This kind of callbacks is used when the engine is being updated and you want to patch
 *  the entries, while their are still under locks - limiting concurrent access and providing
 *  consistency.
 */
struct dummy_callback_t {
    template <typename member_at> void operator()(member_at&&) const noexcept {}
};

/**
 *  @brief  An example of what a USearch-compatible progress-bar should look like.
 *
 *  This is particularly helpful when handling long-running tasks, like serialization,
 *  saving, and loading from disk, or index-level joins.
 *  The reporter checks return value to continue or stop the process, `false` means need to stop.
 */
struct dummy_progress_t {
    inline bool operator()(std::size_t /*processed*/, std::size_t /*total*/) const noexcept { return true; }
};

/**
 *  @brief  An example of what a USearch-compatible values prefetching mechanism should look like.
 *
 *  USearch is designed to handle very large datasets, that may not fir into RAM. Fetching from
 *  external memory is very expensive, so we've added a pre-fetching mechanism, that accepts
 *  multiple objects at once, to cache in RAM ahead of the computation.
 *  The received iterators support both `get_slot` and `get_key` operations.
 *  An example usage may look like this:
 *
 *      template <typename member_citerator_like_at>
 *      inline void operator()(member_citerator_like_at, member_citerator_like_at) const noexcept {
 *          for (; begin != end; ++begin)
 *              io_uring_prefetch(offset_in_file(get_key(begin)));
 *      }
 */
struct dummy_prefetch_t {
    template <typename member_citerator_like_at>
    inline void operator()(member_citerator_like_at, member_citerator_like_at) const noexcept {}
};

/**
 *  @brief  An example of what a USearch-compatible executor (thread-pool) should look like.
 *
 *  It's expected to have `parallel(callback)` API to schedule one task per thread;
 *  an identical `fixed(count, callback)` and `dynamic(count, callback)` overloads that also accepts
 *  the number of tasks, and somehow schedules them between threads; as well as `size()` to
 *  determine the number of available threads.
 */
struct dummy_executor_t {
    dummy_executor_t() noexcept {}
    std::size_t size() const noexcept { return 1; }

    template <typename thread_aware_function_at>
    void fixed(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept {
        for (std::size_t task_idx = 0; task_idx != tasks; ++task_idx)
            thread_aware_function(0, task_idx);
    }

    template <typename thread_aware_function_at>
    void dynamic(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept {
        for (std::size_t task_idx = 0; task_idx != tasks; ++task_idx)
            if (!thread_aware_function(0, task_idx))
                break;
    }

    template <typename thread_aware_function_at>
    void parallel(thread_aware_function_at&& thread_aware_function) noexcept {
        thread_aware_function(0);
    }
};

/**
 *  @brief  An example of what a USearch-compatible key-to-key mapping should look like.
 *
 *  This is particularly helpful for "Semantic Joins", where we map entries of one collection
 *  to entries of another. In asymmetric setups, where A -> B is needed, but B -> A is not,
 *  this can be passed to minimize memory usage.
 */
struct dummy_key_to_key_mapping_t {
    struct member_ref_t {
        template <typename key_at> member_ref_t& operator=(key_at&&) noexcept { return *this; }
    };
    template <typename key_at> member_ref_t operator[](key_at&&) const noexcept { return {}; }
};

/**
 *  @brief  Checks if the provided object has a dummy type, emulating an interface,
 *          but performing no real computation.
 */
template <typename object_at> static constexpr bool is_dummy() {
    using object_t = typename std::remove_all_extents<object_at>::type;
    return std::is_same<typename std::decay<object_t>::type, dummy_predicate_t>::value || //
           std::is_same<typename std::decay<object_t>::type, dummy_callback_t>::value ||  //
           std::is_same<typename std::decay<object_t>::type, dummy_progress_t>::value ||  //
           std::is_same<typename std::decay<object_t>::type, dummy_prefetch_t>::value ||  //
           std::is_same<typename std::decay<object_t>::type, dummy_executor_t>::value ||  //
           std::is_same<typename std::decay<object_t>::type, dummy_key_to_key_mapping_t>::value;
}

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

/**
 *  @brief Smart-pointer wrapping the LibC @b `FILE` for binary file @b outputs.
 *
 * This class raises no exceptions and corresponds errors through `serialization_result_t`.
 * The class automatically closes the file when the object is destroyed.
 */
class output_file_t {
    char const* path_ = nullptr;
    std::FILE* file_ = nullptr;

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
        if (!file_)
            file_ = std::fopen(path_, "wb");
        if (!file_)
            return result.failed(std::strerror(errno));
        return result;
    }
    serialization_result_t write(void const* begin, std::size_t length) noexcept {
        serialization_result_t result;
        std::size_t written = std::fwrite(begin, length, 1, file_);
        if (length && !written)
            return result.failed(std::strerror(errno));
        return result;
    }
    void close() noexcept {
        if (file_)
            std::fclose(exchange(file_, nullptr));
    }
};

/**
 *  @brief  Smart-pointer wrapping the LibC @b `FILE` for binary files @b inputs.
 *
 * This class raises no exceptions and corresponds errors through `serialization_result_t`.
 * The class automatically closes the file when the object is destroyed.
 */
class input_file_t {
    char const* path_ = nullptr;
    std::FILE* file_ = nullptr;

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
        if (!file_)
            file_ = std::fopen(path_, "rb");
        if (!file_)
            return result.failed(std::strerror(errno));
        return result;
    }
    serialization_result_t read(void* begin, std::size_t length) noexcept {
        serialization_result_t result;
        std::size_t read = std::fread(begin, length, 1, file_);
        if (length && !read)
            return result.failed(std::feof(file_) ? "End of file reached!" : std::strerror(errno));
        return result;
    }
    void close() noexcept {
        if (file_)
            std::fclose(exchange(file_, nullptr));
    }

    explicit operator bool() const noexcept { return file_; }
    bool seek_to(std::size_t progress) noexcept { return std::fseek(file_, progress, SEEK_SET) == 0; }
    bool seek_to_end() noexcept { return std::fseek(file_, 0L, SEEK_END) == 0; }
    bool infer_progress(std::size_t& progress) noexcept {
        long int result = std::ftell(file_);
        if (result == -1L)
            return false;
        progress = static_cast<std::size_t>(result);
        return true;
    }
};

/**
 *  @brief  Represents a memory-mapped file or a pre-allocated anonymous memory region.
 *
 *  This class provides a convenient way to memory-map a file and access its contents as a block of
 *  memory. The class handles platform-specific memory-mapping operations on Windows, Linux, and MacOS.
 *  The class automatically closes the file when the object is destroyed.
 */
class memory_mapped_file_t {
    char const* path_{}; /**< The path to the file to be memory-mapped. */
    void* ptr_{};        /**< A pointer to the memory-mapping. */
    size_t length_{};    /**< The length of the memory-mapped file in bytes. */

#if defined(USEARCH_DEFINED_WINDOWS)
    HANDLE file_handle_{};    /**< The file handle on Windows. */
    HANDLE mapping_handle_{}; /**< The mapping handle on Windows. */
#else
    int file_descriptor_{}; /**< The file descriptor on Linux and MacOS. */
#endif

  public:
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    byte_t* data() noexcept { return reinterpret_cast<byte_t*>(ptr_); }
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

    memory_mapped_file_t(byte_t* data, std::size_t length) noexcept : ptr_(data), length_(length) {}

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
        if (!path_ || ptr_)
            return result;

#if defined(USEARCH_DEFINED_WINDOWS)

        HANDLE file_handle =
            CreateFile(path_, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        if (file_handle == INVALID_HANDLE_VALUE)
            return result.failed("Opening file failed!");

        std::size_t file_length = GetFileSize(file_handle, 0);
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
        if (descriptor < 0)
            return result.failed(std::strerror(errno));

        // Estimate the file size
        struct stat file_stat;
        int fstat_status = fstat(descriptor, &file_stat);
        if (fstat_status < 0) {
            ::close(descriptor);
            return result.failed(std::strerror(errno));
        }

        // Map the entire file
        byte_t* file = (byte_t*)mmap(NULL, file_stat.st_size, PROT_READ, MAP_SHARED, descriptor, 0);
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
        if (!path_) {
            ptr_ = nullptr;
            length_ = 0;
            return;
        }
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
    std::uint64_t entry_slot = 0;
};

using default_key_t = std::uint64_t;
using default_slot_t = std::uint32_t;
using default_distance_t = float;

template <typename key_at = default_key_t> struct member_gt {
    key_at key;
    std::size_t slot;
};

template <typename key_at> inline std::size_t get_slot(member_gt<key_at> const& m) noexcept { return m.slot; }
template <typename key_at> inline key_at get_key(member_gt<key_at> const& m) noexcept { return m.key; }

template <typename key_at = default_key_t> struct member_cref_gt {
    misaligned_ref_gt<key_at const> key;
    std::size_t slot;
};

template <typename key_at> inline std::size_t get_slot(member_cref_gt<key_at> const& m) noexcept { return m.slot; }
template <typename key_at> inline key_at get_key(member_cref_gt<key_at> const& m) noexcept { return m.key; }

template <typename key_at = default_key_t> struct member_ref_gt {
    misaligned_ref_gt<key_at> key;
    std::size_t slot;

    inline operator member_cref_gt<key_at>() const noexcept { return {key.ptr(), slot}; }
};

template <typename key_at> inline std::size_t get_slot(member_ref_gt<key_at> const& m) noexcept { return m.slot; }
template <typename key_at> inline key_at get_key(member_ref_gt<key_at> const& m) noexcept { return m.key; }

using level_t = std::int16_t;

struct precomputed_constants_t {
    double inverse_log_connectivity{};
    std::size_t neighbors_bytes{};
    std::size_t neighbors_base_bytes{};
};

// todo:: this is public, but then we make assumptions which are not communicated via this interface
// clean these up later
//
/**
 *  @brief  A loosely-structured handle for every node. One such node is created for every member.
 *          To minimize memory usage and maximize the number of entries per cache-line, it only
 *          stores to pointers. The internal tape starts with a `vector_key_t` @b key, then
 *          a `level_t` for the number of graph @b levels in which this member appears,
 *          then the { `neighbors_count_t`, `compressed_slot_t`, `compressed_slot_t` ... } sequences
 *          for @b each-level.
 */
template <typename key_at, typename slot_at> class node_at {
    byte_t* tape_{};

    inline std::size_t node_neighbors_bytes_(const precomputed_constants_t& pre, node_at node) const noexcept {
        return node_neighbors_bytes_(pre, node.level());
    }
    static inline std::size_t node_neighbors_bytes_(const precomputed_constants_t& pre, level_t level) noexcept {
        return pre.neighbors_base_bytes + pre.neighbors_bytes * level;
    }

  public:
    using vector_key_t = key_at;
    using slot_t = slot_at;
    /**
     *  @brief  Integer for the number of node neighbors at a specific level of the
     *          multi-level graph. It's selected to be `std::uint32_t` to improve the
     *          alignment in most common cases.
     */
    using neighbors_count_t = std::uint32_t;
    using span_bytes_t = span_gt<byte_t>;
    explicit node_at(byte_t* tape) noexcept : tape_(tape) {}
    byte_t* tape() const noexcept { return tape_; }
    /**
     *  @brief  How many bytes of memory are needed to form the "head" of the node.
     */
    static constexpr std::size_t head_size_bytes() { return sizeof(vector_key_t) + sizeof(level_t); }
    byte_t* neighbors_tape() const noexcept { return tape_ + head_size_bytes(); }
    explicit operator bool() const noexcept { return tape_; }

    inline span_bytes_t node_bytes(const precomputed_constants_t& pre) const noexcept {
        return {tape(), node_size_bytes(pre, level())};
    }
    inline std::size_t node_size_bytes(const precomputed_constants_t& pre) noexcept {
        return head_size_bytes() + node_neighbors_bytes_(pre, level());
    }
    static inline std::size_t node_size_bytes(const precomputed_constants_t& pre, level_t level) noexcept {
        return head_size_bytes() + node_neighbors_bytes_(pre, level);
    }

    inline static precomputed_constants_t precompute_(index_config_t const& config) noexcept {
        precomputed_constants_t pre;
        // todo:: ask-Ashot:: inverse_log_connectibity does not relly belong here, but the other two do.
        // maybe we can separate these?
        pre.inverse_log_connectivity = 1.0 / std::log(static_cast<double>(config.connectivity));
        pre.neighbors_bytes = config.connectivity * sizeof(slot_t) + sizeof(neighbors_count_t);
        pre.neighbors_base_bytes = config.connectivity_base * sizeof(slot_t) + sizeof(neighbors_count_t);
        return pre;
    }

    node_at() = default;
    node_at(node_at const&) = default;
    node_at& operator=(node_at const&) = default;

    misaligned_ref_gt<vector_key_t const> ckey() const noexcept { return {tape_}; }
    misaligned_ref_gt<vector_key_t> key() const noexcept { return {tape_}; }
    misaligned_ref_gt<level_t> level() const noexcept { return {tape_ + sizeof(vector_key_t)}; }

    void key(vector_key_t v) noexcept { return misaligned_store<vector_key_t>(tape_, v); }
    void level(level_t v) noexcept { return misaligned_store<level_t>(tape_ + sizeof(vector_key_t), v); }
};

static_assert(std::is_trivially_copy_constructible<node_at<default_key_t, default_slot_t>>::value,
              "Nodes must be light!");
static_assert(std::is_trivially_destructible<node_at<default_key_t, default_slot_t>>::value, "Nodes must be light!");

/**
 *  @brief  Approximate Nearest Neighbors Search @b index-structure using the
 *          Hierarchical Navigable Small World @b (HNSW) graphs algorithm.
 *          If classical containers store @b Key->Value mappings, this one can
 *          be seen as a network of keys, accelerating approximate @b Value~>Key visited_members.
 *
 *  Unlike most implementations, this one is generic anc can be used for any search,
 *  not just within equi-dimensional vectors. Examples range from texts to similar Chess
 *  positions.
 *
 *  @tparam storage_at
 *      The storage provider for index_gt. The index uses the storage_at
 *      API to store and retrieve hnsw index nodes and vectors.
 *      see `dummy_storage_single_threaded` for a minimal storage implementation
 *      and interface reference for storage_at.
 *      NOTE: Storage object is taken by reference. It is the caller's responsibility
 *      to make sure the reference is valid whenever the index is being used
 *
 *  @tparam key_at
 *      The type of primary objects stored in the index.
 *      The values, to which those map, are not managed by the same index structure.
 *
 *  @tparam compressed_slot_at
 *      The smallest unsigned integer type to address indexed elements.
 *      It is used internally to maximize space-efficiency and is generally
 *      up-casted to @b `std::size_t` in public interfaces.
 *      Can be a built-in @b `uint32_t`, `uint64_t`, or our custom @b `uint40_t`.
 *      Which makes the most sense for 4B+ entry indexes.
 *
 *  @tparam dynamic_allocator_at
 *      Dynamic memory allocator for temporary buffers, visits indicators, and
 *      priority queues, needed during construction and traversals of graphs.
 *      The allocated buffers may be uninitialized.
 *
 *
 *  @section Features
 *
 *      - Thread-safe for concurrent construction, search, and updates.
 *      - Doesn't allocate new threads, and reuses the ones its called from.
 *      - Allows storing value externally, managing just the similarity index.
 *      - Joins.

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
 *  the used metric and key types somewhere else.
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
 *  tallest "level" of the graph that it belongs to, the external "key", and the
 *  number of "dimensions" in the vector.
 *
 *  @section Metrics, Predicates and Callbacks
 *
 *
 *  @section Smart References and Iterators
 *
 *      -   `member_citerator_t` and `member_iterator_t` have only slots, no indirections.
 *
 *      -   `member_cref_t` and `member_ref_t` contains the `slot` and a reference
 *          to the key. So it passes through 1 level of visited_members in `nodes_`.
 *          Retrieving the key via `get_key` will cause fetching yet another cache line.
 *
 *      -   `member_gt` contains an already prefetched copy of the key.
 *
 */
template <typename storage_at,                                    //
          typename distance_at = default_distance_t,              //
          typename key_at = default_key_t,                        //
          typename compressed_slot_at = default_slot_t,           //
          typename dynamic_allocator_at = std::allocator<byte_t>> //
class index_gt {
  public:
    using storage_t = storage_at;
    using node_lock_t = typename storage_t::lock_type;
    using distance_t = distance_at;
    using vector_key_t = key_at;
    using key_t = vector_key_t;
    using compressed_slot_t = compressed_slot_at;
    using dynamic_allocator_t = dynamic_allocator_at;
    static_assert(sizeof(vector_key_t) >= sizeof(compressed_slot_t), "Having tiny keys doesn't make sense.");

    using member_cref_t = member_cref_gt<vector_key_t>;
    using member_ref_t = member_ref_gt<vector_key_t>;

    using node_t = node_at<vector_key_t, compressed_slot_t>;
    // using node_t = typename storage_t::node_t;

    template <typename ref_at, typename index_at> class member_iterator_gt {
        using ref_t = ref_at;
        using index_t = index_at;

        friend class index_gt;
        member_iterator_gt() noexcept {}
        member_iterator_gt(index_t* index, std::size_t slot) noexcept : index_(index), slot_(slot) {}

        index_t* index_{};
        std::size_t slot_{};

      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = ref_t;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = ref_t;

        reference operator*() const noexcept { return {index_->storage_->get_node_at(slot_).key(), slot_}; }
        vector_key_t key() const noexcept { return index_->storage_->get_node_at(slot_).key(); }

        friend inline std::size_t get_slot(member_iterator_gt const& it) noexcept { return it.slot_; }
        friend inline vector_key_t get_key(member_iterator_gt const& it) noexcept { return it.key(); }

        member_iterator_gt operator++(int) noexcept { return member_iterator_gt(index_, slot_ + 1); }
        member_iterator_gt operator--(int) noexcept { return member_iterator_gt(index_, slot_ - 1); }
        member_iterator_gt operator+(difference_type d) noexcept { return member_iterator_gt(index_, slot_ + d); }
        member_iterator_gt operator-(difference_type d) noexcept { return member_iterator_gt(index_, slot_ - d); }

        // clang-format off
        member_iterator_gt& operator++() noexcept { slot_ += 1; return *this; }
        member_iterator_gt& operator--() noexcept { slot_ -= 1; return *this; }
        member_iterator_gt& operator+=(difference_type d) noexcept { slot_ += d; return *this; }
        member_iterator_gt& operator-=(difference_type d) noexcept { slot_ -= d; return *this; }
        bool operator==(member_iterator_gt const& other) const noexcept { return index_ == other.index_ && slot_ == other.slot_; }
        bool operator!=(member_iterator_gt const& other) const noexcept { return index_ != other.index_ || slot_ != other.slot_; }
        // clang-format on
    };

    using member_iterator_t = member_iterator_gt<member_ref_t, index_gt>;
    using member_citerator_t = member_iterator_gt<member_cref_t, index_gt const>;

    // STL compatibility:
    using value_type = vector_key_t;
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

    using dynamic_allocator_traits_t = std::allocator_traits<dynamic_allocator_t>;
    using byte_t = typename dynamic_allocator_t::value_type;
    static_assert(           //
        sizeof(byte_t) == 1, //
        "Primary allocator must allocate separate addressable bytes");

    using span_bytes_t = span_gt<byte_t>;

  private:
    /**
     *  @brief  Integer for the number of node neighbors at a specific level of the
     *          multi-level graph. It's selected to be `std::uint32_t` to improve the
     *          alignment in most common cases.
     */
    using neighbors_count_t = std::uint32_t;

    using visits_hash_set_t = growing_hash_set_gt<compressed_slot_t, hash_gt<compressed_slot_t>, dynamic_allocator_t>;

    /// @brief A space-efficient internal data-structure used in graph traversal queues.
    struct candidate_t {
        distance_t distance;
        compressed_slot_t slot;
        inline bool operator<(candidate_t other) const noexcept { return distance < other.distance; }
    };

    using candidates_view_t = span_gt<candidate_t const>;
    using candidates_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<candidate_t>;
    using top_candidates_t = sorted_buffer_gt<candidate_t, std::less<candidate_t>, candidates_allocator_t>;
    using next_candidates_t = max_heap_gt<candidate_t, std::less<candidate_t>, candidates_allocator_t>;

    /**
     *  @brief  A slice of the node's tape, containing a the list of neighbors
     *          for a node in a single graph level. It's pre-allocated to fit
     *          as many neighbors "slots", as may be needed at the target level,
     *          and starts with a single integer `neighbors_count_t` counter.
     */
    class neighbors_ref_t {
        byte_t* tape_;

        static constexpr std::size_t shift(std::size_t i = 0) {
            return sizeof(neighbors_count_t) + sizeof(compressed_slot_t) * i;
        }

      public:
        neighbors_ref_t(byte_t* tape) noexcept : tape_(tape) {}
        misaligned_ptr_gt<compressed_slot_t> begin() noexcept { return tape_ + shift(); }
        misaligned_ptr_gt<compressed_slot_t> end() noexcept { return begin() + size(); }
        misaligned_ptr_gt<compressed_slot_t const> begin() const noexcept { return tape_ + shift(); }
        misaligned_ptr_gt<compressed_slot_t const> end() const noexcept { return begin() + size(); }
        compressed_slot_t operator[](std::size_t i) const noexcept {
            return misaligned_load<compressed_slot_t>(tape_ + shift(i));
        }
        std::size_t size() const noexcept { return misaligned_load<neighbors_count_t>(tape_); }
        void clear() noexcept {
            neighbors_count_t n = misaligned_load<neighbors_count_t>(tape_);
            std::memset(tape_, 0, shift(n));
            // misaligned_store<neighbors_count_t>(tape_, 0);
        }
        void push_back(compressed_slot_t slot) noexcept {
            neighbors_count_t n = misaligned_load<neighbors_count_t>(tape_);
            misaligned_store<compressed_slot_t>(tape_ + shift(n), slot);
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
        visits_hash_set_t visits{};
        std::default_random_engine level_generator{};
        std::size_t iteration_cycles{};
        std::size_t computed_distances_count{};

        template <typename value_at, typename metric_at, typename entry_at> //
        inline distance_t measure(value_at const& first, entry_at const& second, metric_at&& metric) noexcept {
            static_assert( //
                std::is_same<entry_at, member_cref_t>::value || std::is_same<entry_at, member_citerator_t>::value,
                "Unexpected type");

            computed_distances_count++;
            return metric(first, second);
        }

        template <typename metric_at, typename entry_at> //
        inline distance_t measure(entry_at const& first, entry_at const& second, metric_at&& metric) noexcept {
            static_assert( //
                std::is_same<entry_at, member_cref_t>::value || std::is_same<entry_at, member_citerator_t>::value,
                "Unexpected type");

            computed_distances_count++;
            return metric(first, second);
        }
    };

    // todo:: do I have to init this?
    // A: Yes! matters a lot in move constructors!!
    storage_t* storage_{};
    index_config_t config_{};
    index_limits_t limits_{};

    mutable dynamic_allocator_t dynamic_allocator_{};

    precomputed_constants_t pre_{};

    /// @brief  Number of "slots" available for `node_t` objects. Equals to @b `limits_.members`.
    usearch_align_m mutable std::atomic<std::size_t> nodes_capacity_{};

    /// @brief  Number of "slots" already storing non-null nodes.
    usearch_align_m mutable std::atomic<std::size_t> nodes_count_{};

    /// @brief  Controls access to `max_level_` and `entry_slot_`.
    ///         If any thread is updating those values, no other threads can `add()` or `search()`.
    std::mutex global_mutex_{};

    /// @brief  The level of the top-most graph in the index. Grows as the logarithm of size, starts from zero.
    level_t max_level_{};

    /// @brief  The slot in which the only node of the top-level graph is stored.
    std::size_t entry_slot_{};

    using contexts_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<context_t>;

    /// @brief  Array of thread-specific buffers for temporary data.
    mutable buffer_gt<context_t, contexts_allocator_t> contexts_{};

  public:
    std::size_t connectivity() const noexcept { return config_.connectivity; }
    std::size_t capacity() const noexcept { return nodes_capacity_; }
    std::size_t size() const noexcept { return nodes_count_; }
    std::size_t max_level() const noexcept { return nodes_count_ ? static_cast<std::size_t>(max_level_) : 0; }
    index_config_t const& config() const noexcept { return config_; }
    index_limits_t const& limits() const noexcept { return limits_; }
    bool is_immutable() const noexcept { return storage_->is_immutable(); }

    /**
     *  @section Exceptions
     *      Doesn't throw, unless the ::metric's and ::allocators's throw on copy-construction.
     */
    explicit index_gt(      //
        storage_t* storage, //
        index_config_t config = {}, dynamic_allocator_t dynamic_allocator = {}) noexcept
        : storage_(storage), config_(config), limits_(0, 0), dynamic_allocator_(std::move(dynamic_allocator)),
          pre_(node_t::precompute_(config)), nodes_count_(0u), max_level_(-1), entry_slot_(0u), contexts_() {}

    /**
     *  @brief  Clones the structure with the same hyper-parameters, but without contents.
     */
    index_gt fork() noexcept { return index_gt{config_, dynamic_allocator_}; }

    ~index_gt() noexcept {
        reset();
        storage_ = nullptr;
    }

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

    copy_result_t copy(index_copy_config_t config = {}) const noexcept {
        copy_result_t result;
        index_gt& other = result.index;
        other = index_gt(config_, dynamic_allocator_);
        if (!other.reserve(limits_))
            return result.failed("Failed to reserve the contexts");

        // Now all is left - is to allocate new `node_t` instances and populate
        // the `other.nodes_` array into it.

        assert(false);
        // for (std::size_t i = 0; i != nodes_count_; ++i)
        //     other.nodes_[i] = other.node_make_copy_(node_bytes_(nodes_[i]));

        // other.nodes_count_ = nodes_count_.load();
        // other.max_level_ = max_level_;
        // other.entry_slot_ = entry_slot_;

        // This controls nothing for now :)
        (void)config;
        return result;
    }

    member_citerator_t cbegin() const noexcept { return {this, 0}; }
    member_citerator_t cend() const noexcept { return {this, size()}; }
    member_citerator_t begin() const noexcept { return {this, 0}; }
    member_citerator_t end() const noexcept { return {this, size()}; }
    member_iterator_t begin() noexcept { return {this, 0}; }
    member_iterator_t end() noexcept { return {this, size()}; }

    member_ref_t at(std::size_t slot) noexcept { return {storage_->get_node_at(slot).key(), slot}; }
    member_cref_t at(std::size_t slot) const noexcept { return {storage_->get_node_at(slot).ckey(), slot}; }
    member_iterator_t iterator_at(std::size_t slot) noexcept { return {this, slot}; }
    member_citerator_t citerator_at(std::size_t slot) const noexcept { return {this, slot}; }

    dynamic_allocator_t const& dynamic_allocator() const noexcept { return dynamic_allocator_; }

#pragma region Adjusting Configuration

    /**
     *  @brief Erases all the vectors from the index.
     *
     *  Will change `size()` to zero, but will keep the same `capacity()`.
     *  Will keep the number of available threads/contexts the same as it was.
     */
    void clear() noexcept {
        if (storage_)
            storage_->clear();

        nodes_count_ = 0;
        max_level_ = -1;
        entry_slot_ = 0u;
    }

    /**
     *  @brief Erases all members from index, closing files, and returning RAM to OS.
     *
     *  Will change both `size()` and `capacity()` to zero.
     *  Will deallocate all threads/contexts.
     *  If the index is memory-mapped - releases the mapping and the descriptor.
     */
    void reset() noexcept {
        clear();

        if (storage_)
            storage_->reset();
        contexts_ = {};
        limits_ = index_limits_t{0, 0};
        nodes_capacity_ = 0;
    }

    /**
     * @brief replace internal storage pointer with the new one
     */
    void reset_storage(storage_t* storage) { storage_ = storage; }

    /**
     *  @brief  Swaps the underlying memory buffers and thread contexts.
     */
    void swap(index_gt& other) noexcept {
        std::swap(storage_, other.storage_);
        std::swap(config_, other.config_);
        std::swap(limits_, other.limits_);
        std::swap(dynamic_allocator_, other.dynamic_allocator_);
        std::swap(pre_, other.pre_);
        std::swap(max_level_, other.max_level_);
        std::swap(entry_slot_, other.entry_slot_);
        std::swap(contexts_, other.contexts_);

        // Non-atomic parts.
        std::size_t capacity_copy = nodes_capacity_;
        std::size_t count_copy = nodes_count_;
        nodes_capacity_ = other.nodes_capacity_.load();
        nodes_count_ = other.nodes_count_.load();
        other.nodes_capacity_ = capacity_copy;
        other.nodes_count_ = count_copy;
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

        bool storage_reserved = storage_->reserve(limits.members);
        buffer_gt<context_t, contexts_allocator_t> new_contexts(limits.threads());
        if (!new_contexts || !storage_reserved)
            return false;

        limits_ = limits;
        nodes_capacity_ = limits.members;
        contexts_ = std::move(new_contexts);
        return true;
    }

#pragma endregion

#pragma region Construction and Search

    struct add_result_t {
        error_t error{};
        std::size_t new_size{};
        std::size_t visited_members{};
        std::size_t computed_distances{};
        std::size_t slot{};

        explicit operator bool() const noexcept { return !error; }
        add_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /// @brief  Describes a matched search result, augmenting `member_cref_t`
    ///         contents with `distance` to the query object.
    struct match_t {
        member_cref_t member;
        distance_t distance;

        inline match_t() noexcept : member({nullptr, 0}), distance(std::numeric_limits<distance_t>::max()) {}

        inline match_t(member_cref_t member, distance_t distance) noexcept : member(member), distance(distance) {}

        inline match_t(match_t&& other) noexcept
            : member({other.member.key.ptr(), other.member.slot}), distance(other.distance) {}

        inline match_t(match_t const& other) noexcept
            : member({other.member.key.ptr(), other.member.slot}), distance(other.distance) {}

        inline match_t& operator=(match_t const& other) noexcept {
            member.key.reset(other.member.key.ptr());
            member.slot = other.member.slot;
            distance = other.distance;
            return *this;
        }

        inline match_t& operator=(match_t&& other) noexcept {
            member.key.reset(other.member.key.ptr());
            member.slot = other.member.slot;
            distance = other.distance;
            return *this;
        }
    };

    class search_result_t {
        storage_t const* storage_{};
        top_candidates_t const* top_{};

        friend class index_gt;
        inline search_result_t(index_gt const& index, top_candidates_t& top) noexcept
            : storage_(index.storage_), top_(&top) {}

      public:
        /** @brief  Number of search results found. */
        std::size_t count{};
        /** @brief  Number of graph nodes traversed. */
        std::size_t visited_members{};
        /** @brief  Number of times the distances were computed. */
        std::size_t computed_distances{};
        error_t error{};

        inline search_result_t() noexcept {}
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
        inline bool contains(vector_key_t key) const noexcept {
            for (std::size_t i = 0; i != count; ++i)
                if (at(i).member.key == key)
                    return true;
            return false;
        }
        inline match_t at(std::size_t i) const noexcept {
            candidate_t const* top_ordered = top_->data();
            candidate_t candidate = top_ordered[i];
            node_t node = storage_->get_node_at(candidate.slot);
            return {member_cref_t{node.ckey(), candidate.slot}, candidate.distance};
        }
        inline std::size_t merge_into(                 //
            vector_key_t* keys, distance_t* distances, //
            std::size_t old_count, std::size_t max_count) const noexcept {

            std::size_t merged_count = old_count;
            for (std::size_t i = 0; i != count; ++i) {
                match_t result = operator[](i);
                distance_t* merged_end = distances + merged_count;
                std::size_t offset = std::lower_bound(distances, merged_end, result.distance) - distances;
                if (offset == max_count)
                    continue;

                std::size_t count_worse = merged_count - offset - (max_count == merged_count);
                std::memmove(keys + offset + 1, keys + offset, count_worse * sizeof(vector_key_t));
                std::memmove(distances + offset + 1, distances + offset, count_worse * sizeof(distance_t));
                keys[offset] = result.member.key;
                distances[offset] = result.distance;
                merged_count += merged_count != max_count;
            }
            return merged_count;
        }
        inline std::size_t dump_to(vector_key_t* keys, distance_t* distances) const noexcept {
            for (std::size_t i = 0; i != count; ++i) {
                match_t result = operator[](i);
                keys[i] = result.member.key;
                distances[i] = result.distance;
            }
            return count;
        }
        inline std::size_t dump_to(vector_key_t* keys) const noexcept {
            for (std::size_t i = 0; i != count; ++i) {
                match_t result = operator[](i);
                keys[i] = result.member.key;
            }
            return count;
        }
    };

    struct cluster_result_t {
        error_t error{};
        std::size_t visited_members{};
        std::size_t computed_distances{};
        match_t cluster{};

        explicit operator bool() const noexcept { return !error; }
        cluster_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /**
     *  @brief  Inserts a new entry into the index. Thread-safe. Supports @b heterogeneous lookups.
     *          Expects needed capacity to be reserved ahead of time: `size() < capacity()`.
     *
     *  @tparam metric_at
     *      A function responsible for computing the distance @b (dis-similarity) between two objects.
     *      It should be callable into distinctly different scenarios:
     *          - `distance_t operator() (value_at, entry_at)` - from new object to existing entries.
     *          - `distance_t operator() (entry_at, entry_at)` - between existing entries.
     *      Where any possible `entry_at` has both two interfaces: `std::size_t slot()`, `vector_key_t key()`.
     *
     *  @param[in] key External identifier/name/descriptor for the new entry.
     *  @param[in] value Content that will be compared against other entries to index.
     *  @param[in] metric Callable object measuring distance between ::value and present objects.
     *  @param[in] config Configuration options for this specific operation.
     *  @param[in] callback On-success callback, executed while the `member_ref_t` is still under lock.
     */
    template <                                   //
        typename value_at,                       //
        typename metric_at,                      //
        typename callback_at = dummy_callback_t, //
        typename prefetch_at = dummy_prefetch_t  //
        >
    add_result_t add(                                           //
        vector_key_t key, value_at&& value, metric_at&& metric, //
        index_update_config_t config = {},                      //
        callback_at&& callback = callback_at{},                 //
        prefetch_at&& prefetch = prefetch_at{}) usearch_noexcept_m {

        add_result_t result;
        if (is_immutable())
            return result.failed("Can't add to an immutable index");

        // Make sure we have enough local memory to perform this request
        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        next_candidates_t& next = context.next_candidates;
        top.clear();
        next.clear();

        // The top list needs one more slot than the connectivity of the base level
        // for the heuristic, that tries to squeeze one more element into saturated list.
        std::size_t connectivity_max = (std::max)(config_.connectivity_base, config_.connectivity);
        std::size_t top_limit = (std::max)(connectivity_max + 1, config.expansion);
        if (!top.reserve(top_limit))
            return result.failed("Out of memory!");
        if (!next.reserve(config.expansion))
            return result.failed("Out of memory!");

        // Determining how much memory to allocate for the node depends on the target level
        std::unique_lock<std::mutex> new_level_lock(global_mutex_);
        level_t max_level_copy = max_level_;      // Copy under lock
        std::size_t entry_idx_copy = entry_slot_; // Copy under lock
        level_t target_level = choose_random_level_(context.level_generator);

        // Make sure we are not overflowing
        std::size_t capacity = nodes_capacity_.load();
        std::size_t new_slot = nodes_count_.fetch_add(1);
        if (new_slot >= capacity) {
            nodes_count_.fetch_sub(1);
            return result.failed("Reserve capacity ahead of insertions!");
        }

        // Allocate the neighbors
        node_t node = storage_->node_make(key, target_level);
        if (!node) {
            nodes_count_.fetch_sub(1);
            return result.failed("Out of memory!");
        }
        if (target_level <= max_level_copy)
            new_level_lock.unlock();

        storage_->node_store(new_slot, node);
        result.new_size = new_slot + 1;
        result.slot = new_slot;
        callback(at(new_slot));
        node_lock_t new_lock = storage_->node_lock(new_slot);

        // Do nothing for the first element
        if (!new_slot) {
            entry_slot_ = new_slot;
            max_level_ = target_level;
            return result;
        }

        // Pull stats
        result.computed_distances = context.computed_distances_count;
        result.visited_members = context.iteration_cycles;

        connect_node_across_levels_(                                //
            value, metric, prefetch,                                //
            new_slot, entry_idx_copy, max_level_copy, target_level, //
            config, context);

        // Normalize stats
        result.computed_distances = context.computed_distances_count - result.computed_distances;
        result.visited_members = context.iteration_cycles - result.visited_members;

        // Updating the entry point if needed
        if (target_level > max_level_copy) {
            entry_slot_ = new_slot;
            max_level_ = target_level;
        }
        return result;
    }

    /**
     *  @brief  Update an existing entry. Thread-safe. Supports @b heterogeneous lookups.
     *
     *  @tparam metric_at
     *      A function responsible for computing the distance @b (dis-similarity) between two objects.
     *      It should be callable into distinctly different scenarios:
     *          - `distance_t operator() (value_at, entry_at)` - from new object to existing entries.
     *          - `distance_t operator() (entry_at, entry_at)` - between existing entries.
     *      For any possible `entry_at` following interfaces will work:
     *          - `std::size_t get_slot(entry_at const &)`
     *          - `vector_key_t get_key(entry_at const &)`
     *
     *  @param[in] iterator Iterator pointing to an existing entry to be replaced.
     *  @param[in] key External identifier/name/descriptor for the entry.
     *  @param[in] value Content that will be compared against other entries in the index.
     *  @param[in] metric Callable object measuring distance between ::value and present objects.
     *  @param[in] config Configuration options for this specific operation.
     *  @param[in] callback On-success callback, executed while the `member_ref_t` is still under lock.
     */
    template <                                   //
        typename value_at,                       //
        typename metric_at,                      //
        typename callback_at = dummy_callback_t, //
        typename prefetch_at = dummy_prefetch_t  //
        >
    add_result_t update(                        //
        member_iterator_t iterator,             //
        vector_key_t key,                       //
        value_at&& value,                       //
        metric_at&& metric,                     //
        index_update_config_t config = {},      //
        callback_at&& callback = callback_at{}, //
        prefetch_at&& prefetch = prefetch_at{}) usearch_noexcept_m {

        usearch_assert_m(!is_immutable(), "Can't add to an immutable index");
        add_result_t result;
        std::size_t old_slot = iterator.slot_;

        // Make sure we have enough local memory to perform this request
        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        next_candidates_t& next = context.next_candidates;
        top.clear();
        next.clear();

        // The top list needs one more slot than the connectivity of the base level
        // for the heuristic, that tries to squeeze one more element into saturated list.
        std::size_t connectivity_max = (std::max)(config_.connectivity_base, config_.connectivity);
        std::size_t top_limit = (std::max)(connectivity_max + 1, config.expansion);
        if (!top.reserve(top_limit))
            return result.failed("Out of memory!");
        if (!next.reserve(config.expansion))
            return result.failed("Out of memory!");

        node_lock_t new_lock = storage_->node_lock(old_slot);
        node_t node = storage_->get_node_at(old_slot);

        level_t node_level = node.level();
        span_bytes_t node_bytes = node.node_bytes(pre_);
        std::memset(node_bytes.data(), 0, node_bytes.size());
        node.level(node_level);

        // Pull stats
        result.computed_distances = context.computed_distances_count;
        result.visited_members = context.iteration_cycles;

        connect_node_across_levels_(                       //
            value, metric, prefetch,                       //
            old_slot, entry_slot_, max_level_, node_level, //
            config, context);
        node.key(key);

        // Normalize stats
        result.computed_distances = context.computed_distances_count - result.computed_distances;
        result.visited_members = context.iteration_cycles - result.visited_members;
        result.slot = old_slot;

        callback(at(old_slot));
        return result;
    }

    /**
     *  @brief Searches for the closest elements to the given ::query. Thread-safe.
     *
     *  @param[in] query Content that will be compared against other entries in the index.
     *  @param[in] wanted The upper bound for the number of results to return.
     *  @param[in] config Configuration options for this specific operation.
     *  @param[in] predicate Optional filtering predicate for `member_cref_t`.
     *  @return Smart object referencing temporary memory. Valid until next `search()`, `add()`, or `cluster()`.
     */
    template <                                     //
        typename value_at,                         //
        typename metric_at,                        //
        typename predicate_at = dummy_predicate_t, //
        typename prefetch_at = dummy_prefetch_t    //
        >
    search_result_t search(                        //
        value_at&& query,                          //
        std::size_t wanted,                        //
        metric_at&& metric,                        //
        index_search_config_t config = {},         //
        predicate_at&& predicate = predicate_at{}, //
        prefetch_at&& prefetch = prefetch_at{}) const noexcept {

        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        search_result_t result{*this, top};
        if (!nodes_count_)
            return result;

        // Go down the level, tracking only the closest match
        result.computed_distances = context.computed_distances_count;
        result.visited_members = context.iteration_cycles;

        if (config.exact) {
            if (!top.reserve(wanted))
                return result.failed("Out of memory!");
            search_exact_(query, metric, predicate, wanted, context);
        } else {
            next_candidates_t& next = context.next_candidates;
            std::size_t expansion = (std::max)(config.expansion, wanted);
            if (!next.reserve(expansion))
                return result.failed("Out of memory!");
            if (!top.reserve(expansion))
                return result.failed("Out of memory!");

            std::size_t closest_slot = search_for_one_(query, metric, prefetch, entry_slot_, max_level_, 0, context);

            // For bottom layer we need a more optimized procedure
            if (!search_to_find_in_base_(query, metric, predicate, prefetch, closest_slot, expansion, context))
                return result.failed("Out of memory!");
        }

        top.sort_ascending();
        top.shrink(wanted);

        // Normalize stats
        result.computed_distances = context.computed_distances_count - result.computed_distances;
        result.visited_members = context.iteration_cycles - result.visited_members;
        result.count = top.size();
        return result;
    }

    /**
     *  @brief Identifies the closest cluster to the given ::query. Thread-safe.
     *
     *  @param[in] query Content that will be compared against other entries in the index.
     *  @param[in] level The index level to target. Higher means lower resolution.
     *  @param[in] config Configuration options for this specific operation.
     *  @param[in] predicate Optional filtering predicate for `member_cref_t`.
     *  @return Smart object referencing temporary memory. Valid until next `search()`, `add()`, or `cluster()`.
     */
    template <                                     //
        typename value_at,                         //
        typename metric_at,                        //
        typename predicate_at = dummy_predicate_t, //
        typename prefetch_at = dummy_prefetch_t    //
        >
    cluster_result_t cluster(                      //
        value_at&& query,                          //
        std::size_t level,                         //
        metric_at&& metric,                        //
        index_cluster_config_t config = {},        //
        predicate_at&& predicate = predicate_at{}, //
        prefetch_at&& prefetch = prefetch_at{}) const noexcept {

        context_t& context = contexts_[config.thread];
        cluster_result_t result;
        if (!nodes_count_)
            return result.failed("No clusters to identify");

        // Go down the level, tracking only the closest match
        result.computed_distances = context.computed_distances_count;
        result.visited_members = context.iteration_cycles;

        next_candidates_t& next = context.next_candidates;
        std::size_t expansion = config.expansion;
        if (!next.reserve(expansion))
            return result.failed("Out of memory!");

        result.cluster.member =
            at(search_for_one_(query, metric, prefetch, entry_slot_, max_level_, level - 1, context));
        result.cluster.distance = context.measure(query, result.cluster.member, metric);

        // Normalize stats
        result.computed_distances = context.computed_distances_count - result.computed_distances;
        result.visited_members = context.iteration_cycles - result.visited_members;

        (void)predicate;
        return result;
    }

#pragma endregion

#pragma region Metadata

    struct stats_t {
        std::size_t nodes{};
        std::size_t edges{};
        std::size_t max_edges{};
        std::size_t allocated_bytes{};
    };

    stats_t stats() const noexcept {
        stats_t result{};

        for (std::size_t i = 0; i != size(); ++i) {
            node_t node = storage_->get_node_at(i);
            std::size_t max_edges = node.level() * config_.connectivity + config_.connectivity_base;
            std::size_t edges = 0;
            for (level_t level = 0; level <= node.level(); ++level)
                edges += neighbors_(node, level).size();

            ++result.nodes;
            result.allocated_bytes += storage_->node_size_bytes(i);
            result.edges += edges;
            result.max_edges += max_edges;
        }
        return result;
    }

    stats_t stats(std::size_t level) const noexcept {
        stats_t result{};

        std::size_t neighbors_bytes = !level ? pre_.neighbors_base_bytes : pre_.neighbors_bytes;
        for (std::size_t i = 0; i != size(); ++i) {
            node_t node = storage_->get_node_at(i);
            if (static_cast<std::size_t>(node.level()) < level)
                continue;

            ++result.nodes;
            result.edges += neighbors_(node, level).size();
            result.allocated_bytes += node_t::head_size_bytes() + neighbors_bytes;
        }

        std::size_t max_edges_per_node = level ? config_.connectivity_base : config_.connectivity;
        result.max_edges = result.nodes * max_edges_per_node;
        return result;
    }

    stats_t stats(stats_t* stats_per_level, std::size_t max_level) const noexcept {

        std::size_t head_bytes = node_t::head_size_bytes();
        for (std::size_t i = 0; i != size(); ++i) {
            node_t node = storage_->get_node_at(i);

            stats_per_level[0].nodes++;
            stats_per_level[0].edges += neighbors_(node, 0).size();
            stats_per_level[0].allocated_bytes += pre_.neighbors_base_bytes + head_bytes;

            level_t node_level = static_cast<level_t>(node.level());
            for (level_t l = 1; l <= (std::min)(node_level, static_cast<level_t>(max_level)); ++l) {
                stats_per_level[l].nodes++;
                stats_per_level[l].edges += neighbors_(node, l).size();
                stats_per_level[l].allocated_bytes += pre_.neighbors_bytes;
            }
        }
        // The `max_edges` parameter can be inferred from `nodes`
        stats_per_level[0].max_edges = stats_per_level[0].nodes * config_.connectivity_base;
        for (std::size_t l = 1; l <= max_level; ++l)
            stats_per_level[l].max_edges = stats_per_level[l].nodes * config_.connectivity;

        // Aggregate stats across levels
        stats_t result{};
        for (std::size_t l = 0; l <= max_level; ++l)
            result.nodes += stats_per_level[l].nodes,                         //
                result.edges += stats_per_level[l].edges,                     //
                result.allocated_bytes += stats_per_level[l].allocated_bytes, //
                result.max_edges += stats_per_level[l].max_edges;             //

        return result;
    }

    /**
     *  @brief  A relatively accurate lower bound on the amount of memory consumed by the system.
     *          In practice it's error will be below 10%.
     *
     *  @see    `serialized_length` for the length of the binary serialized representation.
     */
    std::size_t memory_usage(std::size_t allocator_entry_bytes = default_allocator_entry_bytes()) const noexcept {
        std::size_t total = 0;
        if (!storage_->is_immutable()) {
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

    std::size_t memory_usage_per_node(level_t level) const noexcept { return node_t::node_size_bytes(pre_, level); }

#pragma endregion

#pragma region Serialization

    /**
     *  @brief  Estimate the binary length (in bytes) of the serialized index.
     */
    std::size_t serialized_length() const noexcept {
        std::size_t neighbors_length = 0;
        for (std::size_t i = 0; i != size(); ++i)
            neighbors_length += node_t::node_size_bytes(pre_, storage_->get_node_at(i).level()) + sizeof(level_t);
        return sizeof(index_serialized_header_t) + neighbors_length;
    }

    /**
     *  @brief  Saves serialized binary index representation to a stream.
     */
    template <typename output_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t save_to_stream(output_callback_at&& output, progress_at&& progress = {}) const noexcept {

        serialization_result_t result;

        // Export some basic metadata
        index_serialized_header_t header;
        header.size = nodes_count_;
        header.connectivity = config_.connectivity;
        header.connectivity_base = config_.connectivity_base;
        header.max_level = max_level_;
        header.entry_slot = entry_slot_;

        return storage_->save_nodes_to_stream(output, header, progress);
    }

    /**
     *  @brief  Symmetric to `save_from_stream`, pulls data from a stream.
     *  Note: assumes storage is properly reset and ready for loading the hnsw graph
     */
    template <typename input_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t load_from_stream(input_callback_at&& input, progress_at&& progress = {}) noexcept {

        serialization_result_t result;

        // Pull basic metadata
        index_serialized_header_t header;
        result = storage_->load_nodes_from_stream(input, header, progress);
        if (!result) {
            reset();
            return result;
        }

        // Submit metadata
        config_.connectivity = header.connectivity;
        config_.connectivity_base = header.connectivity_base;
        pre_ = node_t::precompute_(config_);
        nodes_count_ = header.size;
        max_level_ = static_cast<level_t>(header.max_level);
        entry_slot_ = static_cast<compressed_slot_t>(header.entry_slot);
        // allocate dynamic contexts for queries (storage has already been allocated for the deserialization process)
        index_limits_t limits;
        limits.members = header.size;
        if (!reserve(limits)) {
            reset();
            return result.failed("Out of memory");
        }
        return {};
    }

    template <typename progress_at = dummy_progress_t>
    serialization_result_t save(char const* file_path, progress_at&& progress = {}) const noexcept {
        return save(output_file_t(file_path), std::forward<progress_at>(progress));
    }

    template <typename progress_at = dummy_progress_t>
    serialization_result_t load(char const* file_path, progress_at&& progress = {}) noexcept {
        return load(input_file_t(file_path), std::forward<progress_at>(progress));
    }

    /**
     *  @brief  Saves serialized binary index representation to a file, generally on disk.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t save(output_file_t file, progress_at&& progress = {}) const noexcept {

        serialization_result_t io_result = file.open_if_not();
        if (!io_result)
            return io_result;

        serialization_result_t stream_result = save_to_stream(
            [&](const void* buffer, std::size_t length) {
                io_result = file.write(buffer, length);
                return !!io_result;
            },
            std::forward<progress_at>(progress));

        if (!stream_result)
            return stream_result;
        return io_result;
    }

    /**
     *  @brief  Memory-maps the serialized binary index representation from disk,
     *          @b without copying data into RAM, and fetching it on-demand.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t save(memory_mapped_file_t file, std::size_t offset = 0,
                                progress_at&& progress = {}) const noexcept {

        serialization_result_t io_result = file.open_if_not();
        if (!io_result)
            return io_result;

        serialization_result_t stream_result = save_to_stream(
            [&](const void* buffer, std::size_t length) {
                if (offset + length > file.size())
                    return false;
                std::memcpy(file.data() + offset, buffer, length);
                offset += length;
                return true;
            },
            std::forward<progress_at>(progress));

        return stream_result;
    }

    /**
     *  @brief  Loads the serialized binary index representation from disk to RAM.
     *          Adjusts the configuration properties of the constructed index to
     *          match the settings in the file.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t load(input_file_t file, progress_at&& progress = {}) noexcept {

        serialization_result_t io_result = file.open_if_not();
        if (!io_result)
            return io_result;

        // Remove previously stored objects
        reset();

        serialization_result_t stream_result = load_from_stream(
            [&](void* buffer, std::size_t length) {
                io_result = file.read(buffer, length);
                return !!io_result;
            },
            std::forward<progress_at>(progress));

        if (!stream_result)
            return stream_result;
        return io_result;
    }

    /**
     *  @brief  Loads the serialized binary index representation from disk to RAM.
     *          Adjusts the configuration properties of the constructed index to
     *          match the settings in the file.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t load(memory_mapped_file_t file, std::size_t offset = 0,
                                progress_at&& progress = {}) noexcept {

        serialization_result_t io_result = file.open_if_not();
        if (!io_result)
            return io_result;

        // Remove previously stored objects
        reset();

        serialization_result_t stream_result = load_from_stream(
            [&](void* buffer, std::size_t length) {
                if (offset + length > file.size())
                    return false;
                std::memcpy(buffer, file.data() + offset, length);
                offset += length;
                return true;
            },
            std::forward<progress_at>(progress));

        return stream_result;
    }

    /**
     *  @brief  Memory-maps the serialized binary index representation from disk,
     *          @b without copying data into RAM, and fetching it on-demand.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t view(memory_mapped_file_t file, std::size_t offset = 0,
                                progress_at&& progress = {}) noexcept {

        // Remove previously stored objects
        reset();
        return view_internal(std::move(file), offset, progress);
    }
    template <typename progress_at = dummy_progress_t>
    serialization_result_t view_internal(memory_mapped_file_t file, std::size_t offset = 0,
                                         progress_at&& progress = {}) noexcept {
        // shall not call reset()
        // storage_ may already have some relevant stuff...
        serialization_result_t result;
        index_serialized_header_t header;
        result = storage_->view_nodes_from_file(std::move(file), header, offset, progress);
        if (!result)
            return result;

        config_.connectivity = header.connectivity;
        config_.connectivity_base = header.connectivity_base;
        pre_ = node_t::precompute_(config_);

        // Submit metadata and reserve memory
        index_limits_t limits;
        limits.members = header.size;
        if (!reserve(limits)) {
            reset();
            return result.failed("Out of memory");
        }
        nodes_count_ = header.size;
        max_level_ = static_cast<level_t>(header.max_level);
        entry_slot_ = static_cast<compressed_slot_t>(header.entry_slot);

        return {};
    }

#pragma endregion

  private:
    // todo:: these can also be moved to node_at, along with class neighbors_ref_t definition
    inline neighbors_ref_t neighbors_base_(node_t node) const noexcept { return {node.neighbors_tape()}; }

    inline neighbors_ref_t neighbors_non_base_(node_t node, level_t level) const noexcept {
        return {node.neighbors_tape() + pre_.neighbors_base_bytes + (level - 1) * pre_.neighbors_bytes};
    }

    inline neighbors_ref_t neighbors_(node_t node, level_t level) const noexcept {
        return level ? neighbors_non_base_(node, level) : neighbors_base_(node);
    }

    template <typename value_at, typename metric_at, typename prefetch_at>
    void connect_node_across_levels_(                                                           //
        value_at&& value, metric_at&& metric, prefetch_at&& prefetch,                           //
        std::size_t node_slot, std::size_t entry_slot, level_t max_level, level_t target_level, //
        index_update_config_t const& config, context_t& context) usearch_noexcept_m {

        // Go down the level, tracking only the closest match
        std::size_t closest_slot = search_for_one_( //
            value, metric, prefetch,                //
            entry_slot, max_level, target_level, context);

        // From `target_level` down perform proper extensive search
        for (level_t level = (std::min)(target_level, max_level); level >= 0; --level) {
            // TODO: Handle out of memory conditions
            search_to_insert_(value, metric, prefetch, closest_slot, node_slot, level, config.expansion, context);
            closest_slot = connect_new_node_(metric, node_slot, level, context);
            reconnect_neighbor_nodes_(metric, node_slot, value, level, context);
        }
    }

    template <typename metric_at>
    std::size_t connect_new_node_( //
        metric_at&& metric, std::size_t new_slot, level_t level, context_t& context) usearch_noexcept_m {

        node_t new_node = storage_->get_node_at(new_slot);
        top_candidates_t& top = context.top_candidates;

        // Outgoing links from `new_slot`:
        neighbors_ref_t new_neighbors = neighbors_(new_node, level);
        {
            usearch_assert_m(!new_neighbors.size(), "The newly inserted element should have blank link list");
            candidates_view_t top_view = refine_(metric, config_.connectivity, top, context);

            for (std::size_t idx = 0; idx != top_view.size(); idx++) {
                usearch_assert_m(!new_neighbors[idx], "Possible memory corruption");
                usearch_assert_m(level <= storage_->get_node_at(top_view[idx].slot).level(),
                                 "Linking to missing level");
                new_neighbors.push_back(top_view[idx].slot);
            }
        }

        return new_neighbors[0];
    }

    template <typename value_at, typename metric_at>
    void reconnect_neighbor_nodes_( //
        metric_at&& metric, std::size_t new_slot, value_at&& value, level_t level,
        context_t& context) usearch_noexcept_m {

        node_t new_node = storage_->get_node_at(new_slot);
        top_candidates_t& top = context.top_candidates;
        neighbors_ref_t new_neighbors = neighbors_(new_node, level);

        // Reverse links from the neighbors:
        std::size_t const connectivity_max = level ? config_.connectivity : config_.connectivity_base;
        for (compressed_slot_t close_slot : new_neighbors) {
            if (close_slot == new_slot)
                continue;
            node_lock_t close_lock = storage_->node_lock(close_slot);
            node_t close_node = storage_->get_node_at(close_slot);

            neighbors_ref_t close_header = neighbors_(close_node, level);
            usearch_assert_m(close_header.size() <= connectivity_max, "Possible corruption");
            usearch_assert_m(close_slot != new_slot, "Self-loops are impossible");
            usearch_assert_m(level <= close_node.level(), "Linking to missing level");

            // If `new_slot` is already present in the neighboring connections of `close_slot`
            // then no need to modify any connections or run the heuristics.
            if (close_header.size() < connectivity_max) {
                close_header.push_back(static_cast<compressed_slot_t>(new_slot));
                continue;
            }

            // To fit a new connection we need to drop an existing one.
            top.clear();
            usearch_assert_m((top.reserve(close_header.size() + 1)), "The memory must have been reserved in `add`");
            top.insert_reserved(
                {context.measure(value, citerator_at(close_slot), metric), static_cast<compressed_slot_t>(new_slot)});
            for (compressed_slot_t successor_slot : close_header)
                top.insert_reserved(
                    {context.measure(citerator_at(close_slot), citerator_at(successor_slot), metric), successor_slot});

            // Export the results:
            close_header.clear();
            candidates_view_t top_view = refine_(metric, connectivity_max, top, context);
            for (std::size_t idx = 0; idx != top_view.size(); idx++)
                close_header.push_back(top_view[idx].slot);
        }
    }

    level_t choose_random_level_(std::default_random_engine& level_generator) const noexcept {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -std::log(distribution(level_generator)) * pre_.inverse_log_connectivity;
        return (level_t)r;
    }

    struct candidates_range_t;
    class candidates_iterator_t {
        friend struct candidates_range_t;

        index_gt const& index_;
        neighbors_ref_t neighbors_;
        visits_hash_set_t& visits_;
        std::size_t current_;

        candidates_iterator_t& skip_missing() noexcept {
            if (!visits_.size())
                return *this;
            while (current_ != neighbors_.size()) {
                compressed_slot_t neighbor_slot = neighbors_[current_];
                if (visits_.test(neighbor_slot))
                    current_++;
                else
                    break;
            }
            return *this;
        }

      public:
        using element_t = compressed_slot_t;
        using iterator_category = std::forward_iterator_tag;
        using value_type = element_t;
        using difference_type = std::ptrdiff_t;
        using pointer = misaligned_ptr_gt<element_t>;
        using reference = misaligned_ref_gt<element_t>;

        reference operator*() const noexcept { return slot(); }
        candidates_iterator_t(index_gt const& index, neighbors_ref_t neighbors, visits_hash_set_t& visits,
                              std::size_t progress) noexcept
            : index_(index), neighbors_(neighbors), visits_(visits), current_(progress) {}
        candidates_iterator_t operator++(int) noexcept {
            return candidates_iterator_t(index_, visits_, neighbors_, current_ + 1).skip_missing();
        }
        candidates_iterator_t& operator++() noexcept {
            ++current_;
            skip_missing();
            return *this;
        }
        bool operator==(candidates_iterator_t const& other) noexcept { return current_ == other.current_; }
        bool operator!=(candidates_iterator_t const& other) noexcept { return current_ != other.current_; }

        vector_key_t key() const noexcept { return index_->get_node_at(slot()).key(); }
        compressed_slot_t slot() const noexcept { return neighbors_[current_]; }
        friend inline std::size_t get_slot(candidates_iterator_t const& it) noexcept { return it.slot(); }
        friend inline vector_key_t get_key(candidates_iterator_t const& it) noexcept { return it.key(); }
    };

    struct candidates_range_t {
        index_gt const& index;
        neighbors_ref_t neighbors;
        visits_hash_set_t& visits;

        candidates_iterator_t begin() const noexcept {
            return candidates_iterator_t{index, neighbors, visits, 0}.skip_missing();
        }
        candidates_iterator_t end() const noexcept { return {index, neighbors, visits, neighbors.size()}; }
    };

    template <typename value_at, typename metric_at, typename prefetch_at = dummy_prefetch_t>
    std::size_t search_for_one_(                                      //
        value_at&& query, metric_at&& metric, prefetch_at&& prefetch, //
        std::size_t closest_slot, level_t begin_level, level_t end_level, context_t& context) const noexcept {

        visits_hash_set_t& visits = context.visits;
        visits.clear();

        // Optional prefetching
        if (!is_dummy<prefetch_at>())
            prefetch(citerator_at(closest_slot), citerator_at(closest_slot + 1));

        distance_t closest_dist = context.measure(query, citerator_at(closest_slot), metric);
        for (level_t level = begin_level; level > end_level; --level) {
            bool changed;
            do {
                changed = false;
                node_lock_t closest_lock = storage_->node_lock(closest_slot);
                neighbors_ref_t closest_neighbors = neighbors_non_base_(storage_->get_node_at(closest_slot), level);

                // Optional prefetching
                if (!is_dummy<prefetch_at>()) {
                    candidates_range_t missing_candidates{*this, closest_neighbors, visits};
                    prefetch(missing_candidates.begin(), missing_candidates.end());
                }

                // Actual traversal
                for (compressed_slot_t candidate_slot : closest_neighbors) {
                    distance_t candidate_dist = context.measure(query, citerator_at(candidate_slot), metric);
                    if (candidate_dist < closest_dist) {
                        closest_dist = candidate_dist;
                        closest_slot = candidate_slot;
                        changed = true;
                    }
                }
                context.iteration_cycles++;
            } while (changed);
        }
        return closest_slot;
    }

    /**
     *  @brief  Traverses a layer of a graph, to find the best place to insert a new node.
     *          Locks the nodes in the process, assuming other threads are updating neighbors lists.
     *  @return `true` if procedure succeeded, `false` if run out of memory.
     */
    template <typename value_at, typename metric_at, typename prefetch_at = dummy_prefetch_t>
    bool search_to_insert_(                                           //
        value_at&& query, metric_at&& metric, prefetch_at&& prefetch, //
        std::size_t start_slot, std::size_t new_slot, level_t level, std::size_t top_limit,
        context_t& context) noexcept {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates; // pop min, push
        top_candidates_t& top = context.top_candidates;    // pop max, push

        visits.clear();
        next.clear();
        top.clear();
        if (!visits.reserve(config_.connectivity_base + 1u))
            return false;

        // Optional prefetching
        if (!is_dummy<prefetch_at>())
            prefetch(citerator_at(start_slot), citerator_at(start_slot + 1));

        distance_t radius = context.measure(query, citerator_at(start_slot), metric);
        next.insert_reserved({-radius, static_cast<compressed_slot_t>(start_slot)});
        top.insert_reserved({radius, static_cast<compressed_slot_t>(start_slot)});
        visits.set(start_slot);

        while (!next.empty()) {

            candidate_t candidacy = next.top();
            if ((-candidacy.distance) > radius && top.size() == top_limit)
                break;

            next.pop();
            context.iteration_cycles++;

            compressed_slot_t candidate_slot = candidacy.slot;
            if (new_slot == candidate_slot)
                continue;
            node_t candidate_ref = storage_->get_node_at(candidate_slot);
            node_lock_t candidate_lock = storage_->node_lock(candidate_slot);
            neighbors_ref_t candidate_neighbors = neighbors_(candidate_ref, level);

            // Optional prefetching
            if (!is_dummy<prefetch_at>()) {
                candidates_range_t missing_candidates{*this, candidate_neighbors, visits};
                prefetch(missing_candidates.begin(), missing_candidates.end());
            }

            // Assume the worst-case when reserving memory
            if (!visits.reserve(visits.size() + candidate_neighbors.size()))
                return false;

            for (compressed_slot_t successor_slot : candidate_neighbors) {
                if (visits.set(successor_slot))
                    continue;

                // node_lock_t successor_lock = node_lock_(successor_slot);
                distance_t successor_dist = context.measure(query, citerator_at(successor_slot), metric);
                if (top.size() < top_limit || successor_dist < radius) {
                    // This can substantially grow our priority queue:
                    next.insert({-successor_dist, successor_slot});
                    // This will automatically evict poor matches:
                    top.insert({successor_dist, successor_slot}, top_limit);
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
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at>
    bool search_to_find_in_base_(                                                               //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        std::size_t start_slot, std::size_t expansion, context_t& context) const noexcept {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates; // pop min, push
        top_candidates_t& top = context.top_candidates;    // pop max, push
        std::size_t const top_limit = expansion;

        visits.clear();
        next.clear();
        top.clear();
        if (!visits.reserve(config_.connectivity_base + 1u))
            return false;

        // Optional prefetching
        if (!is_dummy<prefetch_at>())
            prefetch(citerator_at(start_slot), citerator_at(start_slot + 1));

        distance_t radius = context.measure(query, citerator_at(start_slot), metric);
        next.insert_reserved({-radius, static_cast<compressed_slot_t>(start_slot)});
        top.insert_reserved({radius, static_cast<compressed_slot_t>(start_slot)});
        visits.set(start_slot);

        while (!next.empty()) {

            candidate_t candidate = next.top();
            if ((-candidate.distance) > radius)
                break;

            next.pop();
            context.iteration_cycles++;

            neighbors_ref_t candidate_neighbors = neighbors_base_(storage_->get_node_at(candidate.slot));

            // Optional prefetching
            if (!is_dummy<prefetch_at>()) {
                candidates_range_t missing_candidates{*this, candidate_neighbors, visits};
                prefetch(missing_candidates.begin(), missing_candidates.end());
            }

            // Assume the worst-case when reserving memory
            if (!visits.reserve(visits.size() + candidate_neighbors.size()))
                return false;

            for (compressed_slot_t successor_slot : candidate_neighbors) {
                if (visits.set(successor_slot))
                    continue;

                distance_t successor_dist = context.measure(query, citerator_at(successor_slot), metric);
                if (top.size() < top_limit || successor_dist < radius) {
                    // This can substantially grow our priority queue:
                    next.insert({-successor_dist, successor_slot});
                    if (!is_dummy<predicate_at>())
                        if (!predicate(member_cref_t{storage_->get_node_at(successor_slot).ckey(), successor_slot}))
                            continue;

                    // This will automatically evict poor matches:
                    top.insert({successor_dist, successor_slot}, top_limit);
                    radius = top.top().distance;
                }
            }
        }

        return true;
    }

    /**
     *  @brief  Iterates through all members, without actually touching the index.
     */
    template <typename value_at, typename metric_at, typename predicate_at>
    void search_exact_(                                                 //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, //
        std::size_t count, context_t& context) const noexcept {

        top_candidates_t& top = context.top_candidates;
        top.clear();
        top.reserve(count);
        for (std::size_t i = 0; i != size(); ++i) {
            if (!is_dummy<predicate_at>())
                if (!predicate(at(i)))
                    continue;

            distance_t distance = context.measure(query, citerator_at(i), metric);
            top.insert(candidate_t{distance, static_cast<compressed_slot_t>(i)}, count);
        }
    }

    /**
     *  @brief  This algorithm from the original paper implements a heuristic,
     *          that massively reduces the number of connections a point has,
     *          to keep only the neighbors, that are from each other.
     */
    template <typename metric_at>
    candidates_view_t refine_( //
        metric_at&& metric,    //
        std::size_t needed, top_candidates_t& top, context_t& context) const noexcept {

        top.sort_ascending();
        candidate_t* top_data = top.data();
        std::size_t const top_count = top.size();
        if (top_count < needed)
            return {top_data, top_count};

        std::size_t submitted_count = 1;
        std::size_t consumed_count = 1; /// Always equal or greater than `submitted_count`.
        while (submitted_count < needed && consumed_count < top_count) {
            candidate_t candidate = top_data[consumed_count];
            bool good = true;
            for (std::size_t idx = 0; idx < submitted_count; idx++) {
                candidate_t submitted = top_data[idx];
                distance_t inter_result_dist = context.measure( //
                    citerator_at(candidate.slot),               //
                    citerator_at(submitted.slot),               //
                    metric);
                if (inter_result_dist < candidate.distance) {
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

struct join_result_t {
    error_t error{};
    std::size_t intersection_size{};
    std::size_t engagements{};
    std::size_t visited_members{};
    std::size_t computed_distances{};

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
 *  @param[inout] man_to_woman Container to map ::first keys to ::second.
 *  @param[inout] woman_to_man Container to map ::second keys to ::first.
 *  @param[in] executor Thread-pool to execute the job in parallel.
 *  @param[in] progress Callback to report the execution progress.
 */
template < //

    typename men_at,          //
    typename women_at,        //
    typename men_values_at,   //
    typename women_values_at, //
    typename men_metric_at,   //
    typename women_metric_at, //

    typename man_to_woman_at = dummy_key_to_key_mapping_t, //
    typename woman_to_man_at = dummy_key_to_key_mapping_t, //
    typename executor_at = dummy_executor_t,               //
    typename progress_at = dummy_progress_t                //
    >
static join_result_t join(               //
    men_at const& men,                   //
    women_at const& women,               //
    men_values_at const& men_values,     //
    women_values_at const& women_values, //
    men_metric_at&& men_metric,          //
    women_metric_at&& women_metric,      //

    index_join_config_t config = {},                    //
    man_to_woman_at&& man_to_woman = man_to_woman_at{}, //
    woman_to_man_at&& woman_to_man = woman_to_man_at{}, //
    executor_at&& executor = executor_at{},             //
    progress_at&& progress = progress_at{}) noexcept {

    if (women.size() < men.size())
        return unum::usearch::join(                                                               //
            women, men,                                                                           //
            women_values, men_values,                                                             //
            std::forward<women_metric_at>(women_metric), std::forward<men_metric_at>(men_metric), //

            config,                                      //
            std::forward<woman_to_man_at>(woman_to_man), //
            std::forward<man_to_woman_at>(man_to_woman), //
            std::forward<executor_at>(executor),         //
            std::forward<progress_at>(progress));

    join_result_t result;

    // Sanity checks and argument validation:
    if (&men == &women)
        return result.failed("Can't join with itself, consider copying");

    if (config.max_proposals == 0)
        config.max_proposals = std::log(men.size()) + executor.size();

    using proposals_count_t = std::uint16_t;
    config.max_proposals = (std::min)(men.size(), config.max_proposals);

    using distance_t = typename men_at::distance_t;
    using dynamic_allocator_traits_t = typename men_at::dynamic_allocator_traits_t;
    using man_key_t = typename men_at::vector_key_t;
    using woman_key_t = typename women_at::vector_key_t;

    // Use the `compressed_slot_t` type of the larger collection
    using compressed_slot_t = typename women_at::compressed_slot_t;
    using compressed_slot_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<compressed_slot_t>;
    using proposals_count_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<proposals_count_t>;

    // Create an atomic queue, as a ring structure, from/to which
    // free men will be added/pulled.
    std::mutex free_men_mutex{};
    ring_gt<compressed_slot_t, compressed_slot_allocator_t> free_men;
    free_men.reserve(men.size());
    for (std::size_t i = 0; i != men.size(); ++i)
        free_men.push(static_cast<compressed_slot_t>(i));

    // We are gonna need some temporary memory.
    buffer_gt<proposals_count_t, proposals_count_allocator_t> proposal_counts(men.size());
    buffer_gt<compressed_slot_t, compressed_slot_allocator_t> man_to_woman_slots(men.size());
    buffer_gt<compressed_slot_t, compressed_slot_allocator_t> woman_to_man_slots(women.size());
    if (!proposal_counts || !man_to_woman_slots || !woman_to_man_slots)
        return result.failed("Can't temporary mappings");

    compressed_slot_t missing_slot;
    std::memset((void*)&missing_slot, 0xFF, sizeof(compressed_slot_t));
    std::memset((void*)man_to_woman_slots.data(), 0xFF, sizeof(compressed_slot_t) * men.size());
    std::memset((void*)woman_to_man_slots.data(), 0xFF, sizeof(compressed_slot_t) * women.size());
    std::memset(proposal_counts.data(), 0, sizeof(proposals_count_t) * men.size());

    // Define locks, to limit concurrent accesses to `man_to_woman_slots` and `woman_to_man_slots`.
    bitset_t men_locks(men.size()), women_locks(women.size());
    if (!men_locks || !women_locks)
        return result.failed("Can't allocate locks");

    std::atomic<std::size_t> rounds{0};
    std::atomic<std::size_t> engagements{0};
    std::atomic<std::size_t> computed_distances{0};
    std::atomic<std::size_t> visited_members{0};
    std::atomic<char const*> atomic_error{nullptr};

    // Concurrently process all the men
    executor.parallel([&](std::size_t thread_idx) {
        index_search_config_t search_config;
        search_config.expansion = config.expansion;
        search_config.exact = config.exact;
        search_config.thread = thread_idx;
        compressed_slot_t free_man_slot;

        // While there exist a free man who still has a woman to propose to.
        while (!atomic_error.load(std::memory_order_relaxed)) {
            std::size_t passed_rounds = 0;
            std::size_t total_rounds = 0;
            {
                std::unique_lock<std::mutex> pop_lock(free_men_mutex);
                if (!free_men.try_pop(free_man_slot))
                    // Primary exit path, we have exhausted the list of candidates
                    break;
                passed_rounds = ++rounds;
                total_rounds = passed_rounds + free_men.size();
            }
            if (thread_idx == 0 && !progress(passed_rounds, total_rounds)) {
                atomic_error.store("Terminated by user");
                break;
            }
            while (men_locks.atomic_set(free_man_slot))
                ;

            proposals_count_t& free_man_proposals = proposal_counts[free_man_slot];
            if (free_man_proposals >= config.max_proposals)
                continue;

            // Find the closest woman, to whom this man hasn't proposed yet.
            ++free_man_proposals;
            auto candidates = women.search(men_values[free_man_slot], free_man_proposals, women_metric, search_config);
            visited_members += candidates.visited_members;
            computed_distances += candidates.computed_distances;
            if (!candidates) {
                atomic_error = candidates.error.release();
                break;
            }

            auto match = candidates.back();
            auto woman = match.member;
            while (women_locks.atomic_set(woman.slot))
                ;

            compressed_slot_t husband_slot = woman_to_man_slots[woman.slot];
            bool woman_is_free = husband_slot == missing_slot;
            if (woman_is_free) {
                // Engagement
                man_to_woman_slots[free_man_slot] = woman.slot;
                woman_to_man_slots[woman.slot] = free_man_slot;
                engagements++;
            } else {
                distance_t distance_from_husband = women_metric(women_values[woman.slot], men_values[husband_slot]);
                distance_t distance_from_candidate = match.distance;
                if (distance_from_husband > distance_from_candidate) {
                    // Break-up
                    while (men_locks.atomic_set(husband_slot))
                        ;
                    man_to_woman_slots[husband_slot] = missing_slot;
                    men_locks.atomic_reset(husband_slot);

                    // New Engagement
                    man_to_woman_slots[free_man_slot] = woman.slot;
                    woman_to_man_slots[woman.slot] = free_man_slot;
                    engagements++;

                    std::unique_lock<std::mutex> push_lock(free_men_mutex);
                    free_men.push(husband_slot);
                } else {
                    std::unique_lock<std::mutex> push_lock(free_men_mutex);
                    free_men.push(free_man_slot);
                }
            }

            men_locks.atomic_reset(free_man_slot);
            women_locks.atomic_reset(woman.slot);
        }
    });

    if (atomic_error)
        return result.failed(atomic_error.load());

    // Export the "slots" into keys:
    std::size_t intersection_size = 0;
    for (std::size_t man_slot = 0; man_slot != men.size(); ++man_slot) {
        compressed_slot_t woman_slot = man_to_woman_slots[man_slot];
        if (woman_slot != missing_slot) {
            man_key_t man = men.at(man_slot).key;
            woman_key_t woman = women.at(woman_slot).key;
            man_to_woman[man] = woman;
            woman_to_man[woman] = man;
            intersection_size++;
        }
    }

    // Export stats
    result.engagements = engagements;
    result.intersection_size = intersection_size;
    result.computed_distances = computed_distances;
    result.visited_members = visited_members;
    return result;
}

} // namespace usearch
} // namespace unum

#endif
