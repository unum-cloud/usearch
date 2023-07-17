/**
 *  @file index.hpp
 *  @author Ashot Vardanian
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

using f64_t = double;
using f32_t = float;
using byte_t = char;
enum b1x8_t : unsigned char {};

enum class metric_kind_t : std::uint8_t {
    unknown_k = 0,
    // Classics:
    ip_k = 'i',
    cos_k = 'c',
    l2sq_k = 'e',

    // Custom:
    pearson_k = 'p',
    haversine_k = 'h',

    // Sets:
    jaccard_k = 'j',
    hamming_k = 'b',
    tanimoto_k = 't',
    sorensen_k = 's',
};

enum class scalar_kind_t : std::uint8_t {
    unknown_k = 0,
    f64_k,
    f32_k,
    f16_k,
    f8_k,
    b1x8_k,
};

template <typename scalar_at> scalar_kind_t common_scalar_kind() noexcept {
    if (std::is_same<scalar_at, f32_t>())
        return scalar_kind_t::f32_k;
    if (std::is_same<scalar_at, f64_t>())
        return scalar_kind_t::f64_k;
    if (std::is_same<scalar_at, b1x8_t>())
        return scalar_kind_t::b1x8_k;
    return scalar_kind_t::unknown_k;
}

template <typename at> at angle_to_radians(at angle) noexcept { return angle * at(3.14159265358979323846) / at(180); }

template <typename at> at square(at value) noexcept { return value * value; }

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
    v |= v >> 32;
    v++;
    return v;
}

template <typename at> void misaligned_store(void* ptr, at v) noexcept {
    static_assert(!std::is_reference<at>::value);
    std::memcpy(ptr, &v, sizeof(at));
}

/// @brief  Simply dereferencing misaligned pointers can be dangerous.
template <typename at> at misaligned_load(void* ptr) noexcept {
    static_assert(!std::is_reference<at>::value);
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

template <typename at> class misaligned_ptr_gt {
    byte_t* ptr_;

  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = at;
    using difference_type = std::ptrdiff_t;
    using pointer = misaligned_ptr_gt<at>;
    using reference = misaligned_ref_gt<at>;

    reference operator*() const noexcept { return {ptr_}; }

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
 *  @brief  Inner (Dot) Product distance.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct ip_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;

    inline metric_kind_t kind() const noexcept { return metric_kind_t::ip_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t dim) const noexcept {
        result_type ab{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dim; ++i)
            ab += result_t(a[i]) * result_t(b[i]);
        return 1 - ab;
    }
};

/**
 *  @brief  Cosine (Angular) distance.
 *          Identical to the Inner Product of normalized vectors.
 *          Unless you are running on an tiny embedded platform, this metric
 *          is recommended over `::ip_gt` for low-precision scalars.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct cos_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;

    inline metric_kind_t kind() const noexcept { return metric_kind_t::cos_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t dim) const noexcept {
        result_t ab{}, a2{}, b2{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab, a2, b2)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dim; ++i)
            ab += result_t(a[i]) * result_t(b[i]), //
                a2 += square<result_t>(a[i]),      //
                b2 += square<result_t>(b[i]);
        return (ab != 0) ? (1 - ab / (std::sqrt(a2) * std::sqrt(b2))) : 1;
    }
};

/**
 *  @brief  Squared Euclidean (L2) distance.
 *          Square root is avoided at the end, as it won't affect the ordering.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct l2sq_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;

    inline metric_kind_t kind() const noexcept { return metric_kind_t::l2sq_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t dim) const noexcept {
        result_t ab_deltas_sq{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab_deltas_sq)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dim; ++i)
            ab_deltas_sq += square(result_t(a[i]) - result_t(b[i]));
        return ab_deltas_sq;
    }
};

/**
 *  @brief  Hamming distance computes the number of differing bits in
 *          two arrays of integers. An example would be a textual document,
 *          tokenized and hashed into a fixed-capacity bitset.
 */
template <typename scalar_at = std::uint64_t, typename result_at = std::size_t> struct hamming_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;
    static_assert( //
        std::is_unsigned<scalar_t>::value ||
            (std::is_enum<scalar_t>::value && std::is_unsigned<typename std::underlying_type<scalar_t>::type>::value),
        "Hamming distance requires unsigned integral words");

    inline metric_kind_t kind() const noexcept { return metric_kind_t::hamming_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t words) const noexcept {
        constexpr std::size_t bits_per_word_k = sizeof(scalar_t) * CHAR_BIT;
        result_t matches{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : matches)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != words; ++i)
            matches += std::bitset<bits_per_word_k>(a[i] ^ b[i]).count();
        return matches;
    }
};

/**
 *  @brief  Tanimoto distance is the intersection over bitwise union.
 *          Often used in chemistry and biology to compare molecular fingerprints.
 */
template <typename scalar_at = std::uint64_t, typename result_at = float> struct tanimoto_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;
    static_assert( //
        std::is_unsigned<scalar_t>::value ||
            (std::is_enum<scalar_t>::value && std::is_unsigned<typename std::underlying_type<scalar_t>::type>::value),
        "Tanimoto distance requires unsigned integral words");
    static_assert(std::is_floating_point<result_t>::value, "Tanimoto distance will be a fraction");

    inline metric_kind_t kind() const noexcept { return metric_kind_t::tanimoto_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t words) const noexcept {
        constexpr std::size_t bits_per_word_k = sizeof(scalar_t) * CHAR_BIT;
        result_t and_count{};
        result_t or_count{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : and_count, or_count)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != words; ++i)
            and_count += std::bitset<bits_per_word_k>(a[i] & b[i]).count(),
                or_count += std::bitset<bits_per_word_k>(a[i] | b[i]).count();
        return 1 - result_t(and_count) / or_count;
    }
};

/**
 *  @brief  Sorensen-Dice or F1 distance is the intersection over bitwise union.
 *          Often used in chemistry and biology to compare molecular fingerprints.
 */
template <typename scalar_at = std::uint64_t, typename result_at = float> struct sorensen_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;
    static_assert( //
        std::is_unsigned<scalar_t>::value ||
            (std::is_enum<scalar_t>::value && std::is_unsigned<typename std::underlying_type<scalar_t>::type>::value),
        "Sorensen-Dice distance requires unsigned integral words");
    static_assert(std::is_floating_point<result_t>::value, "Sorensen-Dice distance will be a fraction");

    inline metric_kind_t kind() const noexcept { return metric_kind_t::sorensen_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t words) const noexcept {
        constexpr std::size_t bits_per_word_k = sizeof(scalar_t) * CHAR_BIT;
        result_t and_count{};
        result_t any_count{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : and_count, any_count)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != words; ++i)
            and_count += std::bitset<bits_per_word_k>(a[i] & b[i]).count(),
                any_count += std::bitset<bits_per_word_k>(a[i]).count() + std::bitset<bits_per_word_k>(b[i]).count();
        return 1 - 2 * result_t(and_count) / any_count;
    }
};

/**
 *  @brief  Counts the number of matching elements in two unique sorted sets.
 *          Can be used to compute the similarity between two textual documents
 *          using the IDs of tokens present in them.
 *          Similar to `tanimoto_gt` for dense representations.
 */
template <typename scalar_at = std::int32_t, typename result_at = float> struct jaccard_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;
    static_assert(!std::is_floating_point<scalar_t>::value, "Jaccard distance requires integral scalars");

    inline metric_kind_t kind() const noexcept { return metric_kind_t::jaccard_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept {
        return operator()(a.data(), b.data(), a.size(), b.size());
    }

    inline result_t operator()( //
        scalar_t const* a, scalar_t const* b, std::size_t a_length, std::size_t b_length) const noexcept {
        result_t intersection{};
        std::size_t i{};
        std::size_t j{};
        while (i != a_length && j != b_length) {
            intersection += a[i] == b[j];
            i += a[i] < b[j];
            j += a[i] >= b[j];
        }
        return 1 - intersection / (a_length + b_length - intersection);
    }
};

/**
 *  @brief  Counts the number of matching elements in two unique sorted sets.
 *          Can be used to compute the similarity between two textual documents
 *          using the IDs of tokens present in them.
 */
template <typename scalar_at = float, typename result_at = float> struct pearson_correlation_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;

    inline metric_kind_t kind() const noexcept { return metric_kind_t::pearson_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }
    inline result_t operator()(view_t a, view_t b) const noexcept { return operator()(a.data(), b.data(), a.size()); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t dim) const noexcept {
        result_t a_sum{}, b_sum{}, ab_sum{};
        result_t a_sq_sum{}, b_sq_sum{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : a_sum, b_sum, ab_sum, a_sq_sum, b_sq_sum)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dim; ++i) {
            a_sum += result_t(a[i]);
            b_sum += result_t(b[i]);
            ab_sum += result_t(a[i]) * result_t(b[i]);
            a_sq_sum += result_t(a[i]) * result_t(a[i]);
            b_sq_sum += result_t(b[i]) * result_t(b[i]);
        }
        result_t denom = std::sqrt((dim * a_sq_sum - a_sum * a_sum) * (dim * b_sq_sum - b_sum * b_sum));
        result_t corr = (dim * ab_sum - a_sum * b_sum) / denom;
        return -corr;
    }
};

/**
 *  @brief  Haversine distance for the shortest distance between two nodes on
 *          the surface of a 3D sphere, defined with latitude and longitude.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct haversine_gt {
    using scalar_t = scalar_at;
    using view_t = span_gt<scalar_t const>;
    using result_t = result_at;
    using result_type = result_t;
    static_assert(!std::is_integral<scalar_t>::value, "Latitude and longitude must be floating-node");

    inline metric_kind_t kind() const noexcept { return metric_kind_t::haversine_k; }
    inline scalar_kind_t scalar_kind() const noexcept { return common_scalar_kind<scalar_t>(); }

    inline result_t operator()(scalar_t const* a, scalar_t const* b) const noexcept {
        result_t lat_a = a[0], lon_a = a[1];
        result_t lat_b = b[0], lon_b = b[1];

        result_t lat_delta = angle_to_radians<result_t>(lat_b - lat_a);
        result_t lon_delta = angle_to_radians<result_t>(lon_b - lon_a);

        result_t converted_lat_a = angle_to_radians<result_t>(lat_a);
        result_t converted_lat_b = angle_to_radians<result_t>(lat_b);

        result_t x = //
            square(std::sin(lat_delta / 2.f)) +
            std::cos(converted_lat_a) * std::cos(converted_lat_b) * square(std::sin(lon_delta / 2.f));

        return std::atan2(std::sqrt(x), std::sqrt(1.f - x));
    }
};

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
 *          Extends basic functionality with atomic operations.
 */
template <typename allocator_at = std::allocator<char>> class visits_bitset_gt {
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
 */
class usearch_pack_m uint40_t {
    unsigned char octets[5];

  public:
    inline uint40_t() noexcept { std::memset(octets, 0, 5); }
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
        std::uint32_t& n_tail = *reinterpret_cast<std::uint32_t*>(&n);
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
        std::memcpy((char*)&result, octets, 5);
        return result;
    }

    inline uint40_t& operator++() noexcept { return *this += 1u; }

    inline uint40_t operator++(int) noexcept {
        uint40_t old = *this;
        *this += 1u;
        return old;
    }
};

#if defined(USEARCH_DEFINED_WINDOWS)
#pragma pack(pop) // Reset alignment to default
#endif

static_assert(sizeof(uint40_t) == 5, "uint40_t must be exactly 5 bytes");

template <typename element_at, typename allocator_at = std::allocator<element_at>> //
class ring_gt {
  public:
    using element_t = element_at;
    using allocator_t = allocator_at;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

    using value_type = element_t;

  private:
    element_t* elements_;
    std::size_t capacity_;
    std::size_t head_;
    std::size_t tail_;
    bool empty_;
    allocator_t allocator_;

  public:
    explicit ring_gt(allocator_t const& alloc = allocator_t()) noexcept
        : elements_(nullptr), capacity_(0), head_(0), tail_(0), empty_(true), allocator_(alloc) {}

    ring_gt(ring_gt const&) = delete;
    ring_gt& operator=(ring_gt const&) = delete;

    ~ring_gt() noexcept { reset(); }

    bool empty() const noexcept { return empty_; }
    size_t size() const noexcept {
        if (empty_)
            return 0;
        else if (head_ >= tail_)
            return head_ - tail_;
        else
            return capacity_ - (tail_ - head_);
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

    bool resize(std::size_t n) noexcept {
        element_t* elements = allocator_.allocate(n);
        if (!elements)
            return false;
        reset();
        elements_ = elements;
        capacity_ = n;
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
 *  @brief  The "magic" sequence helps infer the type of the file.
 *          USearch indexes start with the "usearch" string.
 */
constexpr char const* default_magic() { return "usearch"; }

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

    /// @brief Parameter controlling the physical layout of vectors
    /// in memory. When using multi-byte scalar types, like `float`,
    /// dereferencing at mis-aligned locations may cause UB. So you
    /// should set it to at least the size of expected floats.
    /// Moreover, when using SIMD-accelerated metrics, you may prefer
    /// to align to SIMD register size for higher performance with
    /// less split-loads.
    std::size_t vector_alignment = 1;
};

struct index_limits_t {
    std::size_t elements = 0;
    std::size_t threads_add = std::thread::hardware_concurrency();
    std::size_t threads_search = std::thread::hardware_concurrency();

    index_limits_t(std::size_t n, std::size_t t) noexcept : elements(n), threads_add(t), threads_search(t) {}
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

    /// @brief Don't copy the ::vector, if it's persisted elsewhere.
    bool store_vector = true;
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

struct copy_config_t {
    bool copy_vectors = true;
};

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

using file_header_t = byte_t[64];

/**
 *  @brief  Serialized binary representations of the USearch index start with metadata.
 *          Metadata is parsed into a `file_head_t`, containing the USearch package version,
 *          and the properties of the index.
 *
 *  It uses: 13 bytes for file versioning, 22 bytes for structural information = 35 bytes.
 *  The following 24 bytes contain binary size of the graph, of the vectors, and the checksum,
 *  leaving 5 bytes at the end vacant.
 */
struct file_head_t {

    // Versioning: 7 + 2 * 3 = 13 bytes
    using magic_t = char[7];
    using version_major_t = std::uint16_t;
    using version_minor_t = std::uint16_t;
    using version_patch_t = std::uint16_t;

    // Structural: 1 * 6 + 8 * 2 = 22 bytes
    using connectivity_t = std::uint8_t;
    using max_level_t = std::uint8_t;
    using vector_alignment_t = std::uint8_t;
    using bytes_per_label_t = std::uint8_t;
    using bytes_per_id_t = std::uint8_t;
    using size_t = std::uint64_t;
    using entry_idx_t = std::uint64_t;

    // Versioning:
    char const* magic;
    misaligned_ref_gt<version_major_t> version_major;
    misaligned_ref_gt<version_minor_t> version_minor;
    misaligned_ref_gt<version_patch_t> version_patch;

    // Structural:
    misaligned_ref_gt<metric_kind_t> metric;
    misaligned_ref_gt<connectivity_t> connectivity;
    misaligned_ref_gt<max_level_t> max_level;
    misaligned_ref_gt<vector_alignment_t> vector_alignment;
    misaligned_ref_gt<bytes_per_label_t> bytes_per_label;
    misaligned_ref_gt<bytes_per_id_t> bytes_per_id;
    misaligned_ref_gt<scalar_kind_t> scalar_kind;
    misaligned_ref_gt<size_t> size;
    misaligned_ref_gt<entry_idx_t> entry_idx;

    // Additional:
    misaligned_ref_gt<size_t> bytes_for_graphs;
    misaligned_ref_gt<size_t> bytes_for_vectors;
    misaligned_ref_gt<size_t> bytes_checksum;

    file_head_t(byte_t* ptr) noexcept
        : magic((char const*)exchange(ptr, ptr + sizeof(magic_t))),
          version_major(exchange(ptr, ptr + sizeof(version_major_t))),
          version_minor(exchange(ptr, ptr + sizeof(version_minor_t))),
          version_patch(exchange(ptr, ptr + sizeof(version_patch_t))),
          metric(exchange(ptr, ptr + sizeof(metric_kind_t))), connectivity(exchange(ptr, ptr + sizeof(connectivity_t))),
          max_level(exchange(ptr, ptr + sizeof(max_level_t))),
          vector_alignment(exchange(ptr, ptr + sizeof(vector_alignment_t))),
          bytes_per_label(exchange(ptr, ptr + sizeof(bytes_per_label_t))),
          bytes_per_id(exchange(ptr, ptr + sizeof(bytes_per_id_t))),
          scalar_kind(exchange(ptr, ptr + sizeof(scalar_kind_t))), size(exchange(ptr, ptr + sizeof(size_t))),
          entry_idx(exchange(ptr, ptr + sizeof(entry_idx_t))), bytes_for_graphs(exchange(ptr, ptr + sizeof(size_t))),
          bytes_for_vectors(exchange(ptr, ptr + sizeof(size_t))), bytes_checksum(exchange(ptr, ptr + sizeof(size_t))) {}
};

struct file_head_result_t {

    using magic_t = file_head_t::magic_t;
    using version_major_t = file_head_t::version_major_t;
    using version_minor_t = file_head_t::version_minor_t;
    using version_patch_t = file_head_t::version_patch_t;

    using connectivity_t = file_head_t::connectivity_t;
    using max_level_t = file_head_t::max_level_t;
    using vector_alignment_t = file_head_t::vector_alignment_t;
    using bytes_per_label_t = file_head_t::bytes_per_label_t;
    using bytes_per_id_t = file_head_t::bytes_per_id_t;
    using size_t = file_head_t::size_t;
    using entry_idx_t = file_head_t::entry_idx_t;

    // Versioning:
    version_major_t version_major;
    version_minor_t version_minor;
    version_patch_t version_patch;

    // Structural:
    metric_kind_t metric;
    connectivity_t connectivity;
    max_level_t max_level;
    vector_alignment_t vector_alignment;
    bytes_per_label_t bytes_per_label;
    bytes_per_id_t bytes_per_id;
    scalar_kind_t scalar_kind;
    size_t size;
    entry_idx_t entry_idx;

    // Additional:
    size_t bytes_for_graphs;
    size_t bytes_for_vectors;
    size_t bytes_checksum;

    error_t error;

    explicit operator bool() const noexcept { return !error; }
    file_head_result_t failed(error_t message) noexcept {
        error = std::move(message);
        return std::move(*this);
    }
};

static_assert(sizeof(file_header_t) == 64, "File header should be exactly 64 bytes");

/// @brief  C++17 and newer version deprecate the `std::result_of`
template <typename metric_at, typename... args_at>
using return_type_gt =
#if defined(USEARCH_DEFINED_CPP17)
    typename std::invoke_result<metric_at, args_at...>::type;
#else
    typename std::result_of<metric_at(args_at...)>::type;
#endif

/// @brief  OS-specific to wrap open file-descriptors/handles.
#if defined(USEARCH_DEFINED_WINDOWS)
struct viewed_file_t {
    HANDLE file_handle{};
    HANDLE mapping_handle{};
    void* ptr{};
    size_t length{};
    explicit operator bool() const noexcept { return mapping_handle != nullptr; }
};
#else
struct viewed_file_t {
    int file_descriptor{};
    void* ptr{};
    size_t length{};
    explicit operator bool() const noexcept { return file_descriptor != 0; }
};
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

/**
 *  @brief  Approximate Nearest Neighbors Search index using the
 *          Hierarchical Navigable Small World graph algorithm.
 *
 *  @section Features
 *      - Thread-safe for concurrent construction.
 *      - Doesn't allocate new threads, and reuses the ones its called from.
 *      - Search for vectors of different dimensionality, if ::`metric_at` supports that.
 *
 *  @tparam metric_at
 *      A function responsible for computing the distance (dis-similarity) between two vectors.
 *      Can have following signatures:
 *          - `distance_t (*) (span_gt<scalar_t const>, span_gt<scalar_t const>)`
 *          - `distance_t (*) (scalar_t const *, scalar_t const *)`
 *
 *  @tparam label_at
 *      The type of unique labels to assign to vectors.
 *
 *  @tparam id_at
 *      The smallest unsigned integer type to address indexed elements.
 *      Can be a built-in `uint32_t`, `uint64_t`, or our custom `uint40_t`.
 *
 *  @tparam allocator_at
 *      Dynamic memory allocator.
 *
 *  @section Usage
 *
 *  @subsection Exceptions
 *
 *  None of the methods throw exceptions in the "Release" compilation mode.
 *  It may only `throw` if your memory ::allocator_at or ::metric_at isn't
 *  safe to copy.
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
 *  @subsection Colocated Nodes
 *
 *  Every node contains some graph-related data, as well as the original vector.
 *  Assuming the vectors can get large, we allow separating the index from the vectors.
 *  They can either be stored in a single continuous block of memory, or in two separate
 *  locations. Vector storage can also be detached from `index_gt` and managed by the user.
 *  To properly serialize and deserialize, use the right `save()`, `load()`,  and `view()`
 *  overloads.
 *
 *  @subsection Missing Functionality
 *
 *  To simplify the implementation, the `index_gt` lacks endpoints to remove existing
 *  vectors. That, however, is solved by `index_punned_dense_gt`, which also adds automatic casting.
 *
 */
template <typename metric_at = ip_gt<float>,            //
          typename label_at = std::size_t,              //
          typename id_at = std::uint32_t,               //
          typename allocator_at = std::allocator<char>, //
          typename point_allocator_at = allocator_at>   //
class index_gt {
  public:
    using metric_t = metric_at;
    using label_t = label_at;
    using id_t = id_at;
    using allocator_t = allocator_at;
    using point_allocator_t = point_allocator_at;
    static_assert(sizeof(label_t) >= sizeof(id_t), "Having tiny labels doesn't make sense.");

    using scalar_t = typename metric_t::scalar_t;
    using vector_view_t = span_gt<scalar_t const>;
    using distance_t = return_type_gt<metric_t, vector_view_t, vector_view_t>;

    struct member_ref_t {
        misaligned_ref_gt<label_t> label;
        vector_view_t vector;
        id_t id;
    };

    struct member_cref_t {
        label_t label;
        vector_view_t vector;
        id_t id;
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
            node_t node = index_->node_with_id_(offset_);
            return {node.label(), node.vector_view(), static_cast<id_t>(offset_)};
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
    using allocator_type = allocator_t;
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
    using dim_t = std::uint32_t;
    using level_t = std::int32_t;

    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    using point_allocator_traits_t = std::allocator_traits<point_allocator_t>;
    static_assert(sizeof(typename point_allocator_traits_t::value_type) == 1,
                  "Allocator must allocate separate addressable bytes");

    /**
     *  @brief  How much larger (number of neighbors per node) will
     *          the base level be compared to other levels.
     */
    static constexpr std::size_t base_level_multiple_() { return 2; }

    /**
     *  @brief  How many bytes of memory are needed to form the "head" of the node.
     */
    static constexpr std::size_t node_head_bytes_() { return sizeof(label_t) + sizeof(dim_t) + sizeof(level_t); }

    using visits_bitset_t = visits_bitset_gt<allocator_t>;

    struct precomputed_constants_t {
        double inverse_log_connectivity{};
        std::size_t connectivity_max_base{};
        std::size_t neighbors_bytes{};
        std::size_t neighbors_base_bytes{};
    };
    struct candidate_t {
        distance_t distance;
        id_t id;
    };
    struct compare_by_distance_t {
        inline bool operator()(candidate_t a, candidate_t b) const noexcept { return a.distance < b.distance; }
    };

    using candidates_view_t = span_gt<candidate_t const>;
    using candidates_allocator_t = typename allocator_traits_t::template rebind_alloc<candidate_t>;
    using top_candidates_t = sorted_buffer_gt<candidate_t, compare_by_distance_t, candidates_allocator_t>;
    using next_candidates_t = max_heap_gt<candidate_t, compare_by_distance_t, candidates_allocator_t>;

    /**
     *  @brief  A loosely-structured handle for every node. One such node is created for every vector.
     *          To minimize memory usage and maximize the number of entries per cache-line, it only
     *          stores to pointers. To reinterpret those into a structured `node_t` more
     *          information is needed from the parent `index_gt`.
     */
    class node_t {
        byte_t* tape_{};
        scalar_t* vector_{};

      public:
        explicit node_t(byte_t* tape, scalar_t* vector) noexcept : tape_(tape), vector_(vector) {}
        byte_t* tape() const noexcept { return tape_; }
        scalar_t* vector() const noexcept { return vector_; }
        byte_t* neighbors_tape() const noexcept { return tape_ + node_head_bytes_(); }

        node_t() = default;
        node_t(node_t const&) = default;
        node_t& operator=(node_t const&) = default;

        misaligned_ref_gt<label_t> label() const noexcept { return {tape_}; }
        misaligned_ref_gt<dim_t> dim() const noexcept { return {tape_ + sizeof(label_t)}; }
        misaligned_ref_gt<level_t> level() const noexcept { return {tape_ + sizeof(label_t) + sizeof(dim_t)}; }

        void label(label_t v) noexcept { return misaligned_store<label_t>(tape_, v); }
        void dim(dim_t v) noexcept { return misaligned_store<dim_t>(tape_ + sizeof(label_t), v); }
        void level(level_t v) noexcept { return misaligned_store<level_t>(tape_ + sizeof(label_t) + sizeof(dim_t), v); }

        operator vector_view_t() const noexcept { return {vector(), dim()}; }
        vector_view_t vector_view() const noexcept { return {vector(), dim()}; }
        explicit operator member_ref_t() noexcept { return {{tape_}, vector_view()}; }
        explicit operator member_cref_t() const noexcept { return {label(), vector_view()}; }
        explicit operator bool() const noexcept { return tape_; }
    };

    /**
     *  @brief  A slice of the node's tape, containing a the list of neighbors
     *          for a node in a single graph level. It's pre-allocated to fit
     *          as many neighbors IDs, as may be needed at the target level,
     *          and starts with a single integer `neighbors_count_t` counter.
     */
    class neighbors_ref_t {
        byte_t* tape_;

        static constexpr std::size_t shift(std::size_t i = 0) { return sizeof(neighbors_count_t) + sizeof(id_t) * i; }

      public:
        neighbors_ref_t(byte_t* tape) noexcept : tape_(tape) {}
        misaligned_ptr_gt<id_t> begin() noexcept { return tape_ + shift(); }
        misaligned_ptr_gt<id_t> end() noexcept { return begin() + size(); }
        misaligned_ptr_gt<id_t const> begin() const noexcept { return tape_ + shift(); }
        misaligned_ptr_gt<id_t const> end() const noexcept { return begin() + size(); }
        id_t operator[](std::size_t i) const noexcept { return misaligned_load<id_t>(tape_ + shift(i)); }
        std::size_t size() const noexcept { return misaligned_load<neighbors_count_t>(tape_); }
        void clear() noexcept { misaligned_store<neighbors_count_t>(tape_, 0); }
        void push_back(id_t id) noexcept {
            neighbors_count_t n = misaligned_load<neighbors_count_t>(tape_);
            misaligned_store<id_t>(tape_ + shift(n), id);
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

        inline distance_t measure(vector_view_t a, vector_view_t b) noexcept {
            measurements_count++;
            return metric(a, b);
        }
    };

    index_config_t config_{};
    index_limits_t limits_{};
    metric_t metric_{};
    mutable allocator_t allocator_{};
    point_allocator_t point_allocator_{};
    precomputed_constants_t pre_{};
    viewed_file_t viewed_file_{};

    usearch_align_m mutable std::atomic<std::size_t> capacity_{};
    usearch_align_m mutable std::atomic<std::size_t> size_{};

    /// @brief  Controls access to `max_level_` and `entry_id_`.
    ///         If any thread is updating those values, no other threads can `add()` or `search()`.
    std::mutex global_mutex_{};
    level_t max_level_{};
    id_t entry_id_{};

    using node_allocator_t = typename allocator_traits_t::template rebind_alloc<node_t>;
    node_t* nodes_{};
    mutable visits_bitset_t nodes_mutexes_{};

    using context_allocator_t = typename allocator_traits_t::template rebind_alloc<context_t>;
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
    explicit index_gt(index_config_t config = {}, metric_t metric = {}, allocator_t allocator = {},
                      point_allocator_t point_allocator = {}) noexcept
        : config_(config), limits_(0, 0), metric_(metric), allocator_(std::move(allocator)),
          point_allocator_(std::move(point_allocator)), pre_(precompute_(config)), size_(0u), max_level_(-1),
          entry_id_(0u), nodes_(nullptr), nodes_mutexes_(), contexts_(nullptr) {}

    /**
     *  @brief  Clones the structure with the same hyper-parameters, but without contents.
     */
    index_gt fork() noexcept { return index_gt{config_, metric_, allocator_}; }

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
        other = index_gt(config_, metric_, allocator_, point_allocator_);
        if (!other.reserve(limits_))
            return result.failed("Failed to reserve the contexts");

        // Now all is left - is to allocate new `node_t` instances and populate
        // the `other.nodes_` array into it.
        for (std::size_t i = 0; i != size_; ++i)
            other.nodes_[i] = other.node_make_copy_(node_bytes_split_(nodes_[i]));

        other.size_ = size_.load();
        other.max_level_ = max_level_;
        other.entry_id_ = entry_id_;
        return result;
    }

    member_citerator_t cbegin() const noexcept { return {this, 0}; }
    member_citerator_t cend() const noexcept { return {this, size()}; }
    member_citerator_t begin() const noexcept { return {this, 0}; }
    member_citerator_t end() const noexcept { return {this, size()}; }
    member_iterator_t begin() noexcept { return {this, 0}; }
    member_iterator_t end() noexcept { return {this, size()}; }

#pragma region Adjusting Configuration

    /**
     *  @brief  Erases all the vectors from the index.
     *          Will change `size()` to zero, but will keep the same `capacity()`.
     *          Will keep the number of available threads/contexts the same as it was.
     */
    void clear() noexcept {
        std::size_t n = size_;
        for (std::size_t i = 0; i != n; ++i)
            node_free_(i);
        size_ = 0;
        max_level_ = -1;
        entry_id_ = 0u;
    }

    /**
     *  @brief  Erases all the vectors from the index, also deallocating the registry.
     *          Will change both `size()` and `capacity()` to zero.
     *          Will deallocate all threads/contexts.
     */
    void reset() noexcept {
        clear();

        if (nodes_)
            node_allocator_t{}.deallocate(exchange(nodes_, nullptr), limits_.elements);
        if (contexts_) {
            for (std::size_t i = 0; i != limits_.threads(); ++i)
                contexts_[i].~context_t();
            context_allocator_t{}.deallocate(exchange(contexts_, nullptr), limits_.threads());
        }
        limits_ = index_limits_t{0, 0};
        capacity_ = 0;
        reset_view_();
    }

    /**
     *  @brief  Swaps the underlying memory buffers and thread contexts.
     */
    void swap(index_gt& other) noexcept {
        std::swap(config_, other.config_);
        std::swap(limits_, other.limits_);
        std::swap(metric_, other.metric_);
        std::swap(allocator_, other.allocator_);
        std::swap(point_allocator_, other.point_allocator_);
        std::swap(pre_, other.pre_);
        std::swap(viewed_file_, other.viewed_file_);
        std::swap(max_level_, other.max_level_);
        std::swap(entry_id_, other.entry_id_);
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

        usearch_assert_m(limits.elements >= size_, "Can't drop existing values");

        if (!nodes_mutexes_.resize(limits.elements))
            return false;

        node_allocator_t node_allocator;
        node_t* new_nodes = node_allocator.allocate(limits.elements);
        if (!new_nodes)
            return false;

        std::size_t limits_threads = limits.threads();
        context_allocator_t context_allocator;
        context_t* new_contexts = context_allocator.allocate(limits_threads);
        if (!new_contexts) {
            node_allocator.deallocate(new_nodes, limits.elements);
            return false;
        }
        for (std::size_t i = 0; i != limits_threads; ++i) {
            context_t& context = new_contexts[i];
            new (&context) context_t();
            context.metric = metric_;
            if (!context.visits.resize(limits.elements)) {
                for (std::size_t j = 0; j != i; ++j)
                    context.visits.reset();
                node_allocator.deallocate(new_nodes, limits.elements);
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
            std::memcpy(new_nodes, nodes_, sizeof(node_t) * size()),
                node_allocator.deallocate(nodes_, limits_.elements);
        if (contexts_)
            context_allocator.deallocate(contexts_, limits_.threads());

        limits_ = limits;
        capacity_ = limits.elements;
        nodes_ = new_nodes;
        contexts_ = new_contexts;
        return true;
    }

    /**
     *  @brief  Optimizes configuration options to fit the maximum number
     *          of neighbors in CPU-cache-aligned buffers.
     */
    static index_config_t optimize(index_config_t config) noexcept {
        precomputed_constants_t pre = precompute_(config);
        std::size_t bytes_per_node_base = node_head_bytes_() + pre.neighbors_base_bytes;
        std::size_t rounded_size = divide_round_up<64>(bytes_per_node_base) * 64;
        std::size_t added_connections = (rounded_size - bytes_per_node_base) / sizeof(id_t);
        config.connectivity = config.connectivity + added_connections / base_level_multiple_();
        return config;
    }

#pragma endregion

#pragma region Construction and Search

    struct add_result_t {
        error_t error{};
        std::size_t new_size{};
        std::size_t cycles{};
        std::size_t measurements{};
        id_t id{};

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
            node_t node = index_.node_with_id_(candidate.id);
            return {member_cref_t{node.label(), node.vector_view(), candidate.id}, candidate.distance};
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
     *  @brief Inserts a new vector into the index. Thread-safe.
     *
     *  @param[in] label External identifier/name/descriptor for the vector.
     *  @param[in] vector Contiguous range of scalars forming a vector view.
     *  @param[in] config Configuration options for this specific operation.
     */
    add_result_t add(label_t label, vector_view_t vector, add_config_t config = {}) usearch_noexcept_m {

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
        std::size_t top_limit = (std::max)(base_level_multiple_() * config_.connectivity + 1, config.expansion);
        if (!top.reserve(top_limit))
            return result.failed("Out of memory!");
        if (!next.reserve(config.expansion))
            return result.failed("Out of memory!");

        // Determining how much memory to allocate for the node depends on the target level
        std::unique_lock<std::mutex> new_level_lock(global_mutex_);
        level_t max_level_copy = max_level_; // Copy under lock
        id_t entry_id_copy = entry_id_;      // Copy under lock
        level_t target_level = choose_random_level_(context.level_generator);
        if (target_level <= max_level_copy)
            new_level_lock.unlock();

        // Allocate the neighbors
        node_t node = node_make_(label, vector, target_level, config.store_vector);
        if (!node)
            return result.failed("Out of memory!");
        std::size_t old_size = size_.fetch_add(1);
        id_t new_id = static_cast<id_t>(old_size);
        nodes_[old_size] = node;
        result.new_size = old_size + 1;
        result.id = new_id;
        node_lock_t new_lock = node_lock_(old_size);

        // Do nothing for the first element
        if (!new_id) {
            entry_id_ = new_id;
            max_level_ = target_level;
            return result;
        }

        // Pull stats
        result.measurements = context.measurements_count;
        result.cycles = context.iteration_cycles;

        // Go down the level, tracking only the closest match
        id_t closest_id = search_for_one_(entry_id_copy, vector, max_level_copy, target_level, context);

        // From `target_level` down perform proper extensive search
        for (level_t level = (std::min)(target_level, max_level_copy); level >= 0; --level) {
            // TODO: Handle out of memory conditions
            search_to_insert_(closest_id, vector, level, config.expansion, context);
            closest_id = connect_new_node_(new_id, level, context);
            reconnect_neighbor_nodes_(new_id, level, context);
        }

        // Normalize stats
        result.measurements = context.measurements_count - result.measurements;
        result.cycles = context.iteration_cycles - result.cycles;

        // Updating the entry point if needed
        if (target_level > max_level_copy) {
            entry_id_ = new_id;
            max_level_ = target_level;
        }
        return result;
    }

    /**
     *  @brief Searches for the closest elements to the given ::query. Thread-safe.
     *
     *  @param[in] query Contiguous range of scalars forming a vector view.
     *  @param[in] wanted The upper bound for the number of results to return.
     *  @param[in] config Configuration options for this specific operation.
     *  @param[in] predicate Optional filtering predicate for `member_cref_t`.
     *  @return Smart object referencing temporary memory. Valid until next `search()` or `add()`.
     */
    template <typename predicate_at = dummy_predicate_t>
    search_result_t search( //
        vector_view_t query, std::size_t wanted, search_config_t config = {},
        predicate_at&& predicate = dummy_predicate_t{}) const noexcept {

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

            id_t closest_id = search_for_one_(entry_id_, query, max_level_, 0, context);
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

    template <typename predicate_at = dummy_predicate_t>
    search_result_t search_around(                          //
        id_t hint, vector_view_t query, std::size_t wanted, //
        search_config_t config = {}, predicate_at&& predicate = dummy_predicate_t{}) const noexcept {

        context_t& context = contexts_[config.thread];
        top_candidates_t& top = context.top_candidates;
        next_candidates_t& next = context.next_candidates;
        search_result_t result{*this, top};

        if (!size_)
            return result;

        std::size_t expansion = (std::max)(config.expansion, wanted);
        if (!next.reserve(expansion))
            return result.failed("Out of memory!");
        if (!top.reserve(expansion))
            return result.failed("Out of memory!");

        // Go down the level, tracking only the closest match
        result.measurements = context.measurements_count;
        result.cycles = context.iteration_cycles;

        search_to_find_in_base_(hint, query, expansion, context, std::forward<predicate_at>(predicate));
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
            node_t node = node_with_id_(i);
            std::size_t max_edges = node.level() * config_.connectivity + base_level_multiple_() * config_.connectivity;
            std::size_t edges = 0;
            for (level_t level = 0; level <= node.level(); ++level)
                edges += neighbors_(node, level).size();

            result.allocated_bytes += node_bytes_(node);
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
            node_t node = node_with_id_(i);
            if (static_cast<std::size_t>(node.level()) < level)
                continue;

            result.edges += neighbors_(node, level).size();
            result.allocated_bytes += node_head_bytes_() + node_vector_bytes_(node) + neighbors_bytes;
        }

        std::size_t max_edges_per_node = level ? config_.connectivity : base_level_multiple_() * config_.connectivity;
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
        total += limits_.elements * sizeof(node_t) + allocator_entry_bytes;

        // Temporary data-structures, proportional to the number of threads:
        total += limits_.threads() * sizeof(context_t) + allocator_entry_bytes * 3;
        return total;
    }

    std::size_t memory_usage_per_node(dim_t dim, level_t level) const noexcept { return node_bytes_(dim, level); }

    void change_metric(metric_t const& m) noexcept {
        metric_ = m;
        for (std::size_t i = 0; i != limits_.threads(); ++i)
            contexts_[i].metric = metric_;
    }

#pragma endregion

#pragma region Serialization

    struct serialization_result_t {
        error_t error;

        explicit operator bool() const noexcept { return !error; }
        serialization_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /**
     *  @brief  Saves serialized binary index representation to disk,
     *          co-locating vectors and neighbors lists.
     *          Available on Linux, MacOS, Windows.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t save(char const* file_path, progress_at&& progress = {}) const noexcept {

        // Make sure we have right to write to that file
        serialization_result_t result;
        std::FILE* file = std::fopen(file_path, "wb");
        if (!file)
            return result.failed(std::strerror(errno));

        // Prepare the header with metadata
        file_header_t state_buffer{};
        std::memset(state_buffer, 0, sizeof(file_header_t));
        file_head_t state{state_buffer};
        std::memcpy(state_buffer, default_magic(), std::strlen(default_magic()));

        // Mark compatibility
        state.version_major = USEARCH_VERSION_MAJOR;
        state.version_minor = USEARCH_VERSION_MINOR;
        state.version_patch = USEARCH_VERSION_PATCH;
        state.metric = metric_.kind();

        // Describe state
        state.connectivity = config_.connectivity;
        state.max_level = max_level_;
        state.vector_alignment = config_.vector_alignment;
        state.bytes_per_label = sizeof(label_t);
        state.bytes_per_id = sizeof(id_t);
        state.scalar_kind = metric_.scalar_kind();
        state.size = size_;
        state.entry_idx = entry_id_;

        // Augment with metadata
        std::size_t graphs_bytes = 0;
        std::size_t vectors_bytes = 0;
        for (std::size_t i = 0; i != state.size; ++i) {
            node_t node = node_with_id_(i);
            std::size_t node_bytes = node_bytes_(node);
            std::size_t node_vector_bytes = node_vector_bytes_(node);
            graphs_bytes += node_bytes - node_vector_bytes;
            vectors_bytes += node_vector_bytes;
        }
        state.bytes_for_graphs = graphs_bytes;
        state.bytes_for_vectors = vectors_bytes;
        state.bytes_checksum = 0;

        // Perform serialization
        auto write_chunk = [&](void* begin, std::size_t length) {
            std::size_t written = std::fwrite(begin, length, 1, file);
            if (!written) {
                std::fclose(file);
                result.failed(std::strerror(errno));
            }
        };

        // Write the header
        write_chunk(&state_buffer[0], sizeof(file_header_t));
        if (result.error)
            return result;

        // Serialize nodes one by one
        for (std::size_t i = 0; i != state.size; ++i) {
            node_t node = node_with_id_(i);
            std::size_t node_bytes = node_bytes_(node);
            std::size_t node_vector_bytes = node_vector_bytes_(node);
            // Dump neighbors and vectors, as vectors may be in a disjoint location
            write_chunk(node.tape(), node_bytes - node_vector_bytes);
            if (result.error)
                return result;
            write_chunk(node.vector(), node_vector_bytes);
            if (result.error)
                return result;
            progress(i, state.size);
        }

        std::fclose(file);
        return {};
    }

    /**
     *  @brief  Loads the serialized binary index representation from disk,
     *          copying both vectors and neighbors lists into RAM.
     *          Available on Linux, MacOS, Windows.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t load(char const* file_path, progress_at&& progress = {}) noexcept {

        serialization_result_t result;
        file_header_t state_buffer{};
        std::FILE* file = std::fopen(file_path, "rb");
        if (!file)
            return result.failed(std::strerror(errno));

        auto read_chunk = [&](void* begin, std::size_t length) {
            std::size_t read = std::fread(begin, length, 1, file);
            if (!read) {
                std::fclose(file);
                result.failed(std::strerror(errno));
            }
        };

        // Read the header
        {
            read_chunk(&state_buffer[0], sizeof(file_header_t));
            if (result.error)
                return result;

            file_head_t state{state_buffer};
            if (state.bytes_per_label != sizeof(label_t)) {
                std::fclose(file);
                return result.failed("Incompatible label type!");
            }
            if (state.bytes_per_id != sizeof(id_t)) {
                std::fclose(file);
                return result.failed("Incompatible ID type!");
            }

            config_.connectivity = state.connectivity;
            config_.vector_alignment = state.vector_alignment;
            pre_ = precompute_(config_);

            index_limits_t limits;
            limits.elements = state.size;
            if (!reserve(limits)) {
                std::fclose(file);
                return result.failed("Out of memory");
            }
            size_ = state.size;
            max_level_ = static_cast<level_t>(state.max_level);
            entry_id_ = static_cast<id_t>(state.entry_idx);
        }

        // Load nodes one by one
        std::size_t const size = size_;
        for (std::size_t i = 0; i != size; ++i) {
            label_t label;
            dim_t dim;
            level_t level;
            read_chunk(&label, sizeof(label));
            if (result.error)
                return result;
            read_chunk(&dim, sizeof(dim));
            if (result.error)
                return result;
            read_chunk(&level, sizeof(level));
            if (result.error)
                return result;

            std::size_t node_bytes = node_bytes_(dim, level);
            node_t node = node_malloc_(dim, level);
            node.label(label);
            node.dim(dim);
            node.level(level);
            read_chunk(node.tape() + node_head_bytes_(), node_bytes - node_head_bytes_());
            if (result.error)
                return result;
            nodes_[i] = node;
            progress(i, size);
        }

        std::fclose(file);
        reset_view_();
        return {};
    }

    /**
     *  @brief  Memory-maps the serialized binary index representation from disk,
     *          @b without copying the vectors and neighbors lists into RAM.
     *          Available on Linux, MacOS, but @b not on Windows.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t view(char const* file_path, progress_at&& progress = {}) noexcept {
        serialization_result_t result;

#if defined(USEARCH_DEFINED_WINDOWS)

        HANDLE file_handle =
            CreateFile(file_path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        if (file_handle == INVALID_HANDLE_VALUE)
            return result.failed("Opening file failed!");

        size_t file_length = GetFileSize(file_handle, 0);
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
        viewed_file_.file_handle = file_handle;
        viewed_file_.mapping_handle = mapping_handle;
        viewed_file_.ptr = file;
        viewed_file_.length = file_length;
#else

#if defined(USEARCH_DEFINED_LINUX)
        int descriptor = open(file_path, O_RDONLY | O_NOATIME);
#else
        int descriptor = open(file_path, O_RDONLY);
#endif

        // Estimate the file size
        struct stat file_stat;
        int fstat_status = fstat(descriptor, &file_stat);
        if (fstat_status < 0) {
            close(descriptor);
            return result.failed(std::strerror(errno));
        }

        // Map the entire file
        byte_t* file = (byte_t*)mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (file == MAP_FAILED) {
            close(descriptor);
            return result.failed(std::strerror(errno));
        }
        viewed_file_.file_descriptor = descriptor;
        viewed_file_.ptr = file;
        viewed_file_.length = file_stat.st_size;
#endif // Platform specific code

        // Read the header
        {
            file_head_t state{file};
            if (state.bytes_per_label != sizeof(label_t)) {
                reset_view_();
                return result.failed("Incompatible label type!");
            }
            if (state.bytes_per_id != sizeof(id_t)) {
                reset_view_();
                return result.failed("Incompatible ID type!");
            }

            config_.connectivity = state.connectivity;
            config_.vector_alignment = state.vector_alignment;
            pre_ = precompute_(config_);

            index_limits_t limits;
            limits.elements = state.size;
            limits.threads_add = 0;
            if (!reserve(limits))
                return result.failed("Out of memory!");

            size_ = state.size;
            max_level_ = static_cast<level_t>(state.max_level);
            entry_id_ = static_cast<id_t>(state.entry_idx);
        }

        // Locate every node packed into file
        std::size_t progress_bytes = sizeof(file_header_t);
        std::size_t const size = size_;
        for (std::size_t i = 0; i != size; ++i) {
            byte_t* tape = (byte_t*)(file + progress_bytes);
            dim_t dim = misaligned_load<dim_t>(tape + sizeof(label_t));
            level_t level = misaligned_load<level_t>(tape + sizeof(label_t) + sizeof(dim_t));

            std::size_t node_bytes = node_bytes_(dim, level);
            std::size_t node_vector_bytes = dim * sizeof(scalar_t);
            nodes_[i] = node_t{tape, (scalar_t*)(tape + node_bytes - node_vector_bytes)};
            progress_bytes += node_bytes;
            progress(i, size);
        }

        return {};
    }

#pragma endregion

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

  private:
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
        using ids_allocator_t = typename allocator_traits_t::template rebind_alloc<id_t>;
        allocator_t& alloc = men.allocator_;

        // Create an atomic queue, as a ring structure, from/to which
        // free men will be added/pulled.
        std::mutex free_men_mutex{};
        ring_gt<id_t, ids_allocator_t> free_men;
        free_men.resize(men.size());
        for (std::size_t i = 0; i != men.size(); ++i)
            free_men.push(static_cast<id_t>(i));

        // We are gonna need some temporary memory.
        proposals_count_t* proposal_counts = (proposals_count_t*)alloc.allocate(sizeof(proposals_count_t) * men.size());
        id_t* man_to_woman_ids = (id_t*)alloc.allocate(sizeof(id_t) * men.size());
        id_t* woman_to_man_ids = (id_t*)alloc.allocate(sizeof(id_t) * women.size());
        if (!proposal_counts || !man_to_woman_ids || !woman_to_man_ids)
            return result.failed("Can't temporary mappings");

        id_t missing_id;
        std::memset((void*)&missing_id, 0xFF, sizeof(id_t));
        std::memset((void*)man_to_woman_ids, 0xFF, sizeof(id_t) * men.size());
        std::memset((void*)woman_to_man_ids, 0xFF, sizeof(id_t) * women.size());
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
            id_t free_man_id;

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

                node_t free_man = men.node_with_id_(free_man_id);
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

                id_t husband_id = woman_to_man_ids[woman.id];
                bool woman_is_free = husband_id == missing_id;
                if (woman_is_free) {
                    // Engagement
                    man_to_woman_ids[free_man_id] = woman.id;
                    woman_to_man_ids[woman.id] = free_man_id;
                    engagements++;
                } else {
                    distance_t distance_from_husband = context.measure(woman.vector, men.node_with_id_(husband_id));
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
            id_t woman_id = man_to_woman_ids[i];
            if (woman_id != missing_id) {
                label_t man = men.node_with_id_(i).label();
                label_t woman = women.node_with_id_(woman_id).label();
                man_to_woman[man] = woman;
                woman_to_man[woman] = man;
                intersection_size++;
            }
        }

        // Deallocate memory
        alloc.deallocate((byte_t*)proposal_counts, sizeof(proposals_count_t) * men.size());
        alloc.deallocate((byte_t*)man_to_woman_ids, sizeof(id_t) * men.size());
        alloc.deallocate((byte_t*)woman_to_man_ids, sizeof(id_t) * women.size());

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

    void reset_view_() noexcept {
        if (!viewed_file_)
            return;
#if defined(USEARCH_DEFINED_WINDOWS)
        UnmapViewOfFile(viewed_file_.ptr);
        CloseHandle(viewed_file_.mapping_handle);
        CloseHandle(viewed_file_.file_handle);

#else
        munmap(viewed_file_.ptr, viewed_file_.length);
        close(viewed_file_.file_descriptor);
#endif
        viewed_file_ = {};
    }

    inline static precomputed_constants_t precompute_(index_config_t const& config) noexcept {
        precomputed_constants_t pre;
        pre.connectivity_max_base = config.connectivity * base_level_multiple_();
        pre.inverse_log_connectivity = 1.0 / std::log(static_cast<double>(config.connectivity));
        pre.neighbors_bytes = config.connectivity * sizeof(id_t) + sizeof(neighbors_count_t);
        pre.neighbors_base_bytes = pre.connectivity_max_base * sizeof(id_t) + sizeof(neighbors_count_t);
        return pre;
    }

    inline std::size_t node_bytes_(node_t node) const noexcept { return node_bytes_(node.dim(), node.level()); }
    inline std::size_t node_bytes_(dim_t dim, level_t level) const noexcept {
        return node_head_bytes_() + pre_.neighbors_base_bytes + pre_.neighbors_bytes * level + sizeof(scalar_t) * dim;
    }

    using span_bytes_t = span_gt<byte_t>;
    struct node_bytes_split_t {
        span_bytes_t tape{};
        span_bytes_t vector{};

        node_bytes_split_t() {}
        node_bytes_split_t(span_bytes_t tape, span_bytes_t vector) noexcept : tape(tape), vector(vector) {}

        std::size_t memory_usage() const noexcept { return tape.size() + vector.size(); }
        bool colocated() const noexcept { return tape.end() == vector.begin(); }
        operator node_t() const noexcept { return node_t{tape.begin(), reinterpret_cast<scalar_t*>(vector.begin())}; }
    };

    inline node_bytes_split_t node_bytes_split_(node_t node) const noexcept {
        std::size_t levels_bytes = pre_.neighbors_base_bytes + pre_.neighbors_bytes * node.level();
        std::size_t bytes_in_tape = node_head_bytes_() + levels_bytes;
        return {{node.tape(), bytes_in_tape}, {(byte_t*)node.vector(), node_vector_bytes_(node)}};
    }

    inline std::size_t node_vector_bytes_(dim_t dim) const noexcept { return dim * sizeof(scalar_t); }
    inline std::size_t node_vector_bytes_(node_t node) const noexcept { return node_vector_bytes_(node.dim()); }

    node_bytes_split_t node_malloc_(dim_t dims_to_store, level_t level) noexcept {

        std::size_t vector_bytes = node_vector_bytes_(dims_to_store);
        std::size_t node_bytes = node_bytes_(dims_to_store, level);
        std::size_t non_vector_bytes = node_bytes - vector_bytes;

        byte_t* data = (byte_t*)point_allocator_.allocate(node_bytes);
        if (!data)
            return node_bytes_split_t{};
        return {{data, non_vector_bytes}, {data + non_vector_bytes, vector_bytes}};
    }

    node_t node_make_(label_t label, vector_view_t vector, level_t level, bool store_vector) noexcept {
        node_bytes_split_t node_bytes = node_malloc_(vector.size() * store_vector, level);
        if (store_vector) {
            std::memset(node_bytes.tape.data(), 0, node_bytes.tape.size());
            std::memcpy(node_bytes.vector.data(), vector.data(), node_bytes.vector.size());
        } else {
            std::memset(node_bytes.tape.data(), 0, node_bytes.memory_usage());
        }
        node_t node = node_bytes;
        node.label(label);
        node.dim(static_cast<dim_t>(vector.size()));
        node.level(level);
        return node;
    }

    node_t node_make_copy_(node_bytes_split_t old_bytes) noexcept {
        if (old_bytes.colocated()) {
            byte_t* data = (byte_t*)point_allocator_.allocate(old_bytes.memory_usage());
            std::memcpy(data, old_bytes.tape.data(), old_bytes.memory_usage());
            return node_t{data, reinterpret_cast<scalar_t*>(data + old_bytes.tape.size())};
        } else {
            node_t old_node = old_bytes;
            node_bytes_split_t node_bytes = node_malloc_(old_node.vector_view().size(), old_node.level());
            std::memcpy(node_bytes.tape.data(), old_bytes.tape.data(), old_bytes.tape.size());
            std::memcpy(node_bytes.vector.data(), old_bytes.vector.data(), old_bytes.vector.size());
            return node_bytes;
        }
    }

    void node_free_(std::size_t id) noexcept {

        if (viewed_file_)
            return;

        node_t& node = nodes_[id];
        std::size_t node_bytes = node_bytes_(node) - node_vector_bytes_(node) * !node_bytes_split_(node).colocated();
        point_allocator_.deallocate(node.tape(), node_bytes);
        node = node_t{};
    }

    inline node_t node_with_id_(std::size_t idx) const noexcept { return nodes_[idx]; }
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

    id_t connect_new_node_(id_t new_id, level_t level, context_t& context) usearch_noexcept_m {

        node_t new_node = node_with_id_(new_id);
        top_candidates_t& top = context.top_candidates;

        // Outgoing links from `new_id`:
        neighbors_ref_t new_neighbors = neighbors_(new_node, level);
        {
            usearch_assert_m(!new_neighbors.size(), "The newly inserted element should have blank link list");
            candidates_view_t top_view = refine_(top, config_.connectivity, context);

            for (std::size_t idx = 0; idx != top_view.size(); idx++) {
                usearch_assert_m(!new_neighbors[idx], "Possible memory corruption");
                usearch_assert_m(level <= node_with_id_(top_view[idx].id).level(), "Linking to missing level");
                new_neighbors.push_back(top_view[idx].id);
            }
        }

        return new_neighbors[0];
    }

    void reconnect_neighbor_nodes_(id_t new_id, level_t level, context_t& context) usearch_noexcept_m {

        node_t new_node = node_with_id_(new_id);
        top_candidates_t& top = context.top_candidates;
        neighbors_ref_t new_neighbors = neighbors_(new_node, level);

        // Reverse links from the neighbors:
        std::size_t const connectivity_max = level ? config_.connectivity : pre_.connectivity_max_base;
        for (id_t close_id : new_neighbors) {
            node_t close_node = node_with_id_(close_id);
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
            for (id_t successor_id : close_header)
                top.insert_reserved({context.measure(node_with_id_(successor_id), close_node), successor_id});

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

    id_t search_for_one_(                       //
        id_t closest_id, vector_view_t query,   //
        level_t begin_level, level_t end_level, //
        context_t& context) const noexcept {

        distance_t closest_dist = context.measure(query, node_with_id_(closest_id));
        for (level_t level = begin_level; level > end_level; --level) {
            bool changed;
            do {
                changed = false;
                node_t closest_node = node_with_id_(closest_id);
                node_lock_t closest_lock = node_lock_(closest_id);
                neighbors_ref_t closest_neighbors = neighbors_non_base_(closest_node, level);
                for (id_t candidate_id : closest_neighbors) {
                    distance_t candidate_dist = context.measure(query, node_with_id_(candidate_id));
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
    bool search_to_insert_( //
        id_t start_id, vector_view_t query, level_t level, std::size_t top_limit, context_t& context) noexcept {

        visits_bitset_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates; // pop min, push
        top_candidates_t& top = context.top_candidates;    // pop max, push

        visits.clear();
        next.clear();
        top.clear();

        distance_t radius = context.measure(query, node_with_id_(start_id));
        next.insert_reserved({-radius, start_id});
        top.insert_reserved({radius, start_id});
        visits.set(start_id);

        while (!next.empty()) {

            candidate_t candidacy = next.top();
            if ((-candidacy.distance) > radius && top.size() == top_limit)
                break;

            next.pop();
            context.iteration_cycles++;

            id_t candidate_id = candidacy.id;
            node_t candidate_ref = node_with_id_(candidate_id);
            node_lock_t candidate_lock = node_lock_(candidate_id);
            neighbors_ref_t candidate_neighbors = neighbors_(candidate_ref, level);

            prefetch_neighbors_(candidate_neighbors, visits);
            for (id_t successor_id : candidate_neighbors) {
                if (visits.test(successor_id))
                    continue;

                visits.set(successor_id);
                distance_t successor_dist = context.measure(query, node_with_id_(successor_id));

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
    template <typename predicate_at>
    bool search_to_find_in_base_( //
        id_t start_id, vector_view_t query, std::size_t expansion, context_t& context,
        predicate_at&& predicate) const noexcept {

        visits_bitset_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates; // pop min, push
        top_candidates_t& top = context.top_candidates;    // pop max, push
        std::size_t const top_limit = expansion;

        visits.clear();
        next.clear();
        top.clear();

        distance_t radius = context.measure(query, node_with_id_(start_id));
        next.insert_reserved({-radius, start_id});
        top.insert_reserved({radius, start_id});
        visits.set(start_id);

        while (!next.empty()) {

            candidate_t candidate = next.top();
            if ((-candidate.distance) > radius)
                break;

            next.pop();
            context.iteration_cycles++;

            id_t candidate_id = candidate.id;
            neighbors_ref_t candidate_neighbors = neighbors_base_(node_with_id_(candidate_id));

            prefetch_neighbors_(candidate_neighbors, visits);
            for (id_t successor_id : candidate_neighbors) {
                if (visits.test(successor_id))
                    continue;

                visits.set(successor_id);
                node_t successor = node_with_id_(successor_id);
                distance_t successor_dist = context.measure(query, successor);

                if (top.size() < top_limit || successor_dist < radius) {
                    // This can substantially grow our priority queue:
                    next.insert({-successor_dist, successor_id});
                    if (predicate( //
                            match_t{member_cref_t{successor.label(), successor.vector_view(), successor_id},
                                    successor_dist})) {
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
     *  @brief  Iterates through all managed vectors, without actually touching the index.
     */
    template <typename predicate_at>
    void search_exact_(                                             //
        vector_view_t query, std::size_t count, context_t& context, //
        predicate_at&& predicate) const noexcept {

        top_candidates_t& top = context.top_candidates;
        top.clear();
        top.reserve(count);
        for (std::size_t i = 0; i != size(); ++i) {
            id_t id = static_cast<id_t>(i);
            node_t node = node_with_id_(i);
            distance_t distance = context.measure(query, node);
            if (predicate(match_t{member_cref_t{node.label(), node.vector_view(), id}, distance}))
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
            node_t candidate_ref = node_with_id_(candidate.id);
            distance_t candidate_dist = candidate.distance;
            bool good = true;
            for (std::size_t idx = 0; idx < submitted_count; idx++) {
                candidate_t submitted = top_data[idx];
                node_t submitted_node = node_with_id_(submitted.id);
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
    result.metric = state.metric;
    result.connectivity = state.connectivity;
    result.max_level = state.max_level;
    result.vector_alignment = state.vector_alignment;
    result.bytes_per_label = state.bytes_per_label;
    result.bytes_per_id = state.bytes_per_id;
    result.scalar_kind = state.scalar_kind;
    result.size = state.size;
    result.entry_idx = state.entry_idx;
    result.bytes_for_graphs = state.bytes_for_graphs;
    result.bytes_for_vectors = state.bytes_for_vectors;
    result.bytes_checksum = state.bytes_checksum;
    return result;
}

} // namespace usearch
} // namespace unum

#endif
