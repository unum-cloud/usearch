#pragma once
#include <stdlib.h> // `aligned_alloc`

#include <cstring> // `std::strncmp`
#include <numeric> // `std::iota`
#include <thread>  // `std::thread`
#include <vector>  // `std::vector`

#include <atomic> // `std::atomic`
#include <thread> // `std::thread`

#include <usearch/index.hpp> // `expected_gt` and macros

#if USEARCH_USE_OPENMP
#include <omp.h> // `omp_get_num_threads()`
#endif

#if defined(USEARCH_DEFINED_LINUX)
#include <sys/auxv.h> // `getauxval()`
#endif

#if defined(USEARCH_DEFINED_ARM)
#include <arm_fp16.h> // `__fp16`
#endif

#if !defined(USEARCH_USE_NATIVE_F16)
#if defined(__AVX512F__)
#define USEARCH_USE_NATIVE_F16 1
#elif defined(USEARCH_DEFINED_ARM)
#define USEARCH_USE_NATIVE_F16 1
#else
#define USEARCH_USE_NATIVE_F16 0
#include <fp16/fp16.h>
#endif
#else
#include <fp16/fp16.h>
#endif

#if !defined(USEARCH_USE_SIMSIMD)
#define USEARCH_USE_SIMSIMD 0
#endif

#if USEARCH_USE_SIMSIMD
#include <simsimd/simsimd.h>
#endif

namespace unum {
namespace usearch {

using u40_t = uint40_t;
enum b1x8_t : unsigned char {};

struct uuid_t {
    std::uint8_t octets[16];
};

class f16_bits_t;
class i8_converted_t;

#if USEARCH_USE_NATIVE_F16
#if defined(USEARCH_DEFINED_ARM)
using f16_native_t = __fp16;
#else
using f16_native_t = _Float16;
#endif
using f16_t = f16_native_t;
#else
using f16_native_t = void;
using f16_t = f16_bits_t;
#endif

using f64_t = double;
using f32_t = float;

using u64_t = std::uint64_t;
using u32_t = std::uint32_t;
using u16_t = std::uint16_t;
using u8_t = std::uint8_t;

using i64_t = std::int64_t;
using i32_t = std::int32_t;
using i16_t = std::int16_t;
using i8_t = std::int8_t;

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
    // Custom:
    b1x8_k,
    u40_k,
    uuid_k,
    // Common:
    f64_k,
    f32_k,
    f16_k,
    f8_k,
    // Common Integral:
    u64_k,
    u32_k,
    u16_k,
    u8_k,
    i64_k,
    i32_k,
    i16_k,
    i8_k,
};

enum class isa_kind_t {
    auto_k,
    neon_k,
    sve_k,
    avx2_k,
    avx512_k,
};

enum class prefetching_kind_t {
    none_k,
    cpu_k,
    io_uring_k,
};

template <typename scalar_at> scalar_kind_t scalar_kind() noexcept {
    if (std::is_same<scalar_at, b1x8_t>())
        return scalar_kind_t::b1x8_k;
    if (std::is_same<scalar_at, uint40_t>())
        return scalar_kind_t::u40_k;
    if (std::is_same<scalar_at, uuid_t>())
        return scalar_kind_t::uuid_k;
    if (std::is_same<scalar_at, f64_t>())
        return scalar_kind_t::f64_k;
    if (std::is_same<scalar_at, f32_t>())
        return scalar_kind_t::f32_k;
    if (std::is_same<scalar_at, f16_t>())
        return scalar_kind_t::f16_k;
    if (std::is_same<scalar_at, i8_t>())
        return scalar_kind_t::i8_k;
    if (std::is_same<scalar_at, u64_t>())
        return scalar_kind_t::u64_k;
    if (std::is_same<scalar_at, u32_t>())
        return scalar_kind_t::u32_k;
    if (std::is_same<scalar_at, u16_t>())
        return scalar_kind_t::u16_k;
    if (std::is_same<scalar_at, u8_t>())
        return scalar_kind_t::u8_k;
    if (std::is_same<scalar_at, i64_t>())
        return scalar_kind_t::i64_k;
    if (std::is_same<scalar_at, i32_t>())
        return scalar_kind_t::i32_k;
    if (std::is_same<scalar_at, i16_t>())
        return scalar_kind_t::i16_k;
    if (std::is_same<scalar_at, i8_t>())
        return scalar_kind_t::i8_k;
    return scalar_kind_t::unknown_k;
}

template <typename at> at angle_to_radians(at angle) noexcept { return angle * at(3.14159265358979323846) / at(180); }

template <typename at> at square(at value) noexcept { return value * value; }

template <typename at, typename compare_at> inline at clamp(at v, at lo, at hi, compare_at comp) noexcept {
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
template <typename at> inline at clamp(at v, at lo, at hi) noexcept {
    return usearch::clamp(v, lo, hi, std::less<at>{});
}

inline bool str_equals(char const* begin, std::size_t len, char const* other_begin) noexcept {
    std::size_t other_len = std::strlen(other_begin);
    return len == other_len && std::strncmp(begin, other_begin, len) == 0;
}

inline char const* isa_name(isa_kind_t isa_kind) noexcept {
    switch (isa_kind) {
    case isa_kind_t::auto_k: return "auto";
    case isa_kind_t::neon_k: return "neon";
    case isa_kind_t::sve_k: return "sve";
    case isa_kind_t::avx2_k: return "avx2";
    case isa_kind_t::avx512_k: return "avx512";
    default: return "";
    }
}

inline bool hardware_supports(isa_kind_t isa_kind) noexcept {

    // On Linux Arm machines the `getauxval` can be queried to check
    // if SVE extensions are available. Arm Neon has no separate capability check.
#if defined(USEARCH_DEFINED_ARM) && defined(USEARCH_DEFINED_LINUX)
    unsigned long capabilities = getauxval(AT_HWCAP);
    switch (isa_kind) {
    case isa_kind_t::neon_k: return true; // Must be supported on 64-bit Arm
    case isa_kind_t::sve_k: return capabilities & HWCAP_SVE;
    default: return false;
    }
#endif

    // When compiling with GCC, one may use the "built-ins", including ones
    // designed for CPU capability detection.
#if defined(USEARCH_DEFINED_X86) && defined(USEARCH_DEFINED_GCC)
    __builtin_cpu_init();
    switch (isa_kind) {
    case isa_kind_t::avx2_k: return __builtin_cpu_supports("avx2");
    case isa_kind_t::avx512_k: return __builtin_cpu_supports("avx512f");
    default: return false;
    }
#endif

    // On Apple we can expect Arm devices to support Neon extesions,
    // and the x86 machines to support AVX2 extensions.
#if defined(USEARCH_DEFINED_APPLE)
    switch (isa_kind) {
#if defined(USEARCH_DEFINED_ARM)
    case isa_kind_t::neon_k: return true;
#else
    case isa_kind_t::avx2_k: return true;
#endif
    default: return false;
    }
#endif

    (void)isa_kind;
    return false;
}

inline std::size_t bits_per_scalar(scalar_kind_t scalar_kind) noexcept {
    switch (scalar_kind) {
    case scalar_kind_t::f64_k: return 64;
    case scalar_kind_t::f32_k: return 32;
    case scalar_kind_t::f16_k: return 16;
    case scalar_kind_t::i8_k: return 8;
    case scalar_kind_t::b1x8_k: return 1;
    default: return 0;
    }
}

inline std::size_t bits_per_scalar_word(scalar_kind_t scalar_kind) noexcept {
    switch (scalar_kind) {
    case scalar_kind_t::f64_k: return 64;
    case scalar_kind_t::f32_k: return 32;
    case scalar_kind_t::f16_k: return 16;
    case scalar_kind_t::i8_k: return 8;
    case scalar_kind_t::b1x8_k: return 8;
    default: return 0;
    }
}

inline char const* scalar_kind_name(scalar_kind_t scalar_kind) noexcept {
    switch (scalar_kind) {
    case scalar_kind_t::f32_k: return "f32";
    case scalar_kind_t::f16_k: return "f16";
    case scalar_kind_t::f64_k: return "f64";
    case scalar_kind_t::i8_k: return "i8";
    case scalar_kind_t::b1x8_k: return "b1x8";
    default: return "";
    }
}

inline char const* metric_kind_name(metric_kind_t metric) noexcept {
    switch (metric) {
    case metric_kind_t::unknown_k: return "unknown";
    case metric_kind_t::ip_k: return "ip";
    case metric_kind_t::cos_k: return "cos";
    case metric_kind_t::l2sq_k: return "l2sq";
    case metric_kind_t::pearson_k: return "pearson";
    case metric_kind_t::haversine_k: return "haversine";
    case metric_kind_t::jaccard_k: return "jaccard";
    case metric_kind_t::hamming_k: return "hamming";
    case metric_kind_t::tanimoto_k: return "tanimoto";
    case metric_kind_t::sorensen_k: return "sorensen";
    }
    return "";
}
inline expected_gt<scalar_kind_t> scalar_kind_from_name(char const* name, std::size_t len) {
    expected_gt<scalar_kind_t> parsed;
    if (str_equals(name, len, "f32"))
        parsed.result = scalar_kind_t::f32_k;
    else if (str_equals(name, len, "f64"))
        parsed.result = scalar_kind_t::f64_k;
    else if (str_equals(name, len, "f16"))
        parsed.result = scalar_kind_t::f16_k;
    else if (str_equals(name, len, "i8"))
        parsed.result = scalar_kind_t::i8_k;
    else
        parsed.failed("Unknown type, choose: f32, f16, f64, i8");
    return parsed;
}

inline expected_gt<metric_kind_t> metric_from_name(char const* name, std::size_t len) {
    expected_gt<metric_kind_t> parsed;
    if (str_equals(name, len, "l2sq") || str_equals(name, len, "euclidean_sq")) {
        parsed.result = metric_kind_t::l2sq_k;
    } else if (str_equals(name, len, "ip") || str_equals(name, len, "inner") || str_equals(name, len, "dot")) {
        parsed.result = metric_kind_t::ip_k;
    } else if (str_equals(name, len, "cos") || str_equals(name, len, "angular")) {
        parsed.result = metric_kind_t::cos_k;
    } else if (str_equals(name, len, "haversine")) {
        parsed.result = metric_kind_t::haversine_k;
    } else if (str_equals(name, len, "pearson")) {
        parsed.result = metric_kind_t::pearson_k;
    } else if (str_equals(name, len, "hamming")) {
        parsed.result = metric_kind_t::hamming_k;
    } else if (str_equals(name, len, "tanimoto")) {
        parsed.result = metric_kind_t::tanimoto_k;
    } else if (str_equals(name, len, "sorensen")) {
        parsed.result = metric_kind_t::sorensen_k;
    } else
        parsed.failed(
            "Unknown distance, choose: l2sq, ip, cos, haversine, jaccard, pearson, hamming, tanimoto, sorensen");
    return parsed;
}

inline float f16_to_f32(std::uint16_t u16) noexcept {
#if USEARCH_USE_NATIVE_F16
    f16_native_t f16;
    std::memcpy(&f16, &u16, sizeof(std::uint16_t));
    return float(f16);
#else
    return fp16_ieee_to_fp32_value(u16);
#endif
}

inline std::uint16_t f32_to_f16(float f32) noexcept {
#if USEARCH_USE_NATIVE_F16
    f16_native_t f16 = f16_native_t(f32);
    std::uint16_t u16;
    std::memcpy(&u16, &f16, sizeof(std::uint16_t));
    return u16;
#else
    return fp16_ieee_from_fp32_value(f32);
#endif
}

/**
 *  @brief  Numeric type for the IEEE 754 half-precision floating point.
 *          If hardware support isn't available, falls back to a hardware
 *          agnostic in-software implementation.
 */
class f16_bits_t {
    std::uint16_t uint16_{};

  public:
    inline f16_bits_t() noexcept : uint16_(0) {}
    inline f16_bits_t(f16_bits_t&&) = default;
    inline f16_bits_t& operator=(f16_bits_t&&) = default;
    inline f16_bits_t(f16_bits_t const&) = default;
    inline f16_bits_t& operator=(f16_bits_t const&) = default;

    inline operator float() const noexcept { return f16_to_f32(uint16_); }
    inline explicit operator bool() const noexcept { return f16_to_f32(uint16_) > 0.5f; }

    inline f16_bits_t(i8_converted_t) noexcept;
    inline f16_bits_t(bool v) noexcept : uint16_(f32_to_f16(v)) {}
    inline f16_bits_t(float v) noexcept : uint16_(f32_to_f16(v)) {}
    inline f16_bits_t(double v) noexcept : uint16_(f32_to_f16(v)) {}

    inline f16_bits_t operator+(f16_bits_t other) const noexcept { return {float(*this) + float(other)}; }
    inline f16_bits_t operator-(f16_bits_t other) const noexcept { return {float(*this) - float(other)}; }
    inline f16_bits_t operator*(f16_bits_t other) const noexcept { return {float(*this) * float(other)}; }
    inline f16_bits_t operator/(f16_bits_t other) const noexcept { return {float(*this) / float(other)}; }
    inline f16_bits_t operator+(float other) const noexcept { return {float(*this) + other}; }
    inline f16_bits_t operator-(float other) const noexcept { return {float(*this) - other}; }
    inline f16_bits_t operator*(float other) const noexcept { return {float(*this) * other}; }
    inline f16_bits_t operator/(float other) const noexcept { return {float(*this) / other}; }
    inline f16_bits_t operator+(double other) const noexcept { return {float(*this) + other}; }
    inline f16_bits_t operator-(double other) const noexcept { return {float(*this) - other}; }
    inline f16_bits_t operator*(double other) const noexcept { return {float(*this) * other}; }
    inline f16_bits_t operator/(double other) const noexcept { return {float(*this) / other}; }

    inline f16_bits_t& operator+=(float v) noexcept {
        uint16_ = f32_to_f16(v + f16_to_f32(uint16_));
        return *this;
    }

    inline f16_bits_t& operator-=(float v) noexcept {
        uint16_ = f32_to_f16(v - f16_to_f32(uint16_));
        return *this;
    }

    inline f16_bits_t& operator*=(float v) noexcept {
        uint16_ = f32_to_f16(v * f16_to_f32(uint16_));
        return *this;
    }

    inline f16_bits_t& operator/=(float v) noexcept {
        uint16_ = f32_to_f16(v / f16_to_f32(uint16_));
        return *this;
    }
};

/**
 *  @brief  An STL-based executor or a "thread-pool" for parallel execution.
 *          Isn't efficient for small batches, as it recreates the threads on every call.
 */
class executor_stl_t {
    std::size_t threads_count_{};

    struct jthread_t {
        std::thread native_;

        jthread_t() = default;
        jthread_t(jthread_t&&) = default;
        jthread_t(jthread_t const&) = delete;
        template <typename callable_at> jthread_t(callable_at&& func) : native_([=]() { func(); }) {}

        ~jthread_t() {
            if (native_.joinable())
                native_.join();
        }
    };

  public:
    /**
     *  @param threads_count The number of threads to be used for parallel execution.
     */
    executor_stl_t(std::size_t threads_count = 0) noexcept
        : threads_count_(threads_count ? threads_count : std::thread::hardware_concurrency()) {}

    /**
     *  @return Maximum number of threads available to the executor.
     */
    std::size_t size() const noexcept { return threads_count_; }

    /**
     *  @brief Executes a fixed number of tasks using the specified thread-aware function.
     *  @param tasks                 The total number of tasks to be executed.
     *  @param thread_aware_function The thread-aware function to be called for each thread index and task index.
     *  @throws If an exception occurs during execution of the thread-aware function.
     */
    template <typename thread_aware_function_at>
    void fixed(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept(false) {
        std::vector<jthread_t> threads_pool;
        std::size_t tasks_per_thread = tasks;
        std::size_t threads_count = (std::min)(threads_count_, tasks);
        if (threads_count > 1) {
            tasks_per_thread = (tasks / threads_count) + ((tasks % threads_count) != 0);
            for (std::size_t thread_idx = 1; thread_idx < threads_count; ++thread_idx) {
                threads_pool.emplace_back([=]() {
                    for (std::size_t task_idx = thread_idx * tasks_per_thread;
                         task_idx < (std::min)(tasks, thread_idx * tasks_per_thread + tasks_per_thread); ++task_idx)
                        thread_aware_function(thread_idx, task_idx);
                });
            }
        }
        for (std::size_t task_idx = 0; task_idx < (std::min)(tasks, tasks_per_thread); ++task_idx)
            thread_aware_function(0, task_idx);
    }

    /**
     *  @brief Executes limited number of tasks using the specified thread-aware function.
     *  @param tasks                 The upper bound on the number of tasks.
     *  @param thread_aware_function The thread-aware function to be called for each thread index and task index.
     *  @throws If an exception occurs during execution of the thread-aware function.
     */
    template <typename thread_aware_function_at>
    void dynamic(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept(false) {
        std::vector<jthread_t> threads_pool;
        std::size_t tasks_per_thread = tasks;
        std::size_t threads_count = (std::min)(threads_count_, tasks);
        std::atomic_bool stop{false};
        if (threads_count > 1) {
            tasks_per_thread = (tasks / threads_count) + ((tasks % threads_count) != 0);
            for (std::size_t thread_idx = 1; thread_idx < threads_count; ++thread_idx) {
                threads_pool.emplace_back([=, &stop]() {
                    for (std::size_t task_idx = thread_idx * tasks_per_thread;
                         task_idx < (std::min)(tasks, thread_idx * tasks_per_thread + tasks_per_thread) &&
                         !stop.load(std::memory_order_relaxed);
                         ++task_idx)
                        if (!thread_aware_function(thread_idx, task_idx))
                            stop.store(true, std::memory_order_relaxed);
                });
            }
        }
        for (std::size_t task_idx = 0;
             task_idx < (std::min)(tasks, tasks_per_thread) && !stop.load(std::memory_order_relaxed); ++task_idx)
            if (!thread_aware_function(0, task_idx))
                stop.store(true, std::memory_order_relaxed);
    }

    /**
     *  @brief Saturates every available thread with the given workload, until they finish.
     *  @param thread_aware_function The thread-aware function to be called for each thread index.
     *  @throws If an exception occurs during execution of the thread-aware function.
     */
    template <typename thread_aware_function_at>
    void parallel(thread_aware_function_at&& thread_aware_function) noexcept(false) {
        if (threads_count_ == 1)
            return thread_aware_function(0);
        std::vector<jthread_t> threads_pool;
        for (std::size_t thread_idx = 1; thread_idx < threads_count_; ++thread_idx)
            threads_pool.emplace_back([=]() { thread_aware_function(thread_idx); });
        thread_aware_function(0);
    }
};

#if USEARCH_USE_OPENMP

/**
 *  @brief  An OpenMP-based executor or a "thread-pool" for parallel execution.
 *          Is the preferred implementation, when available, and maximum performance is needed.
 */
class executor_openmp_t {
  public:
    /**
     *  @param threads_count The number of threads to be used for parallel execution.
     */
    executor_openmp_t(std::size_t threads_count = 0) noexcept {
        omp_set_num_threads(threads_count ? threads_count : std::thread::hardware_concurrency());
    }

    /**
     *  @return Maximum number of threads available to the executor.
     */
    std::size_t size() const noexcept { return omp_get_num_threads(); }

    /**
     *  @brief Executes tasks in bulk using the specified thread-aware function.
     *  @param tasks                 The total number of tasks to be executed.
     *  @param thread_aware_function The thread-aware function to be called for each thread index and task index.
     *  @throws If an exception occurs during execution of the thread-aware function.
     */
    template <typename thread_aware_function_at>
    void fixed(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept(false) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t i = 0; i != tasks; ++i) {
            thread_aware_function(omp_get_thread_num(), i);
        }
    }

    /**
     *  @brief Executes tasks in bulk using the specified thread-aware function.
     *  @param tasks                 The total number of tasks to be executed.
     *  @param thread_aware_function The thread-aware function to be called for each thread index and task index.
     *  @throws If an exception occurs during execution of the thread-aware function.
     */
    template <typename thread_aware_function_at>
    void dynamic(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept(false) {
        // OpenMP cancellation points are not yet available on most platforms, and require
        // the `OMP_CANCELLATION` environment variable to be set.
        // http://jakascorner.com/blog/2016/08/omp-cancel.html
        // if (omp_get_cancellation()) {
        // #pragma omp parallel for schedule(dynamic, 1)
        //     for (std::size_t i = 0; i != tasks; ++i) {
        // #pragma omp cancellation point for
        //         if (!thread_aware_function(omp_get_thread_num(), i)) {
        // #pragma omp cancel for
        //         }
        //     }
        // }
        std::atomic_bool stop{false};
#pragma omp parallel for schedule(dynamic, 1) shared(stop)
        for (std::size_t i = 0; i != tasks; ++i) {
            if (!stop.load(std::memory_order_relaxed) && !thread_aware_function(omp_get_thread_num(), i))
                stop.store(true, std::memory_order_relaxed);
        }
    }

    /**
     *  @brief Saturates every available thread with the given workload, until they finish.
     *  @param thread_aware_function The thread-aware function to be called for each thread index.
     *  @throws If an exception occurs during execution of the thread-aware function.
     */
    template <typename thread_aware_function_at>
    void parallel(thread_aware_function_at&& thread_aware_function) noexcept(false) {
#pragma omp parallel
        { thread_aware_function(omp_get_thread_num()); }
    }
};

using executor_default_t = executor_openmp_t;

#else

using executor_default_t = executor_stl_t;

#endif

/**
 *  @brief Uses OS-specific APIs for aligned memory allocations.
 */
template <typename element_at = char, std::size_t alignment_ak = 64> //
class aligned_allocator_gt {
  public:
    using value_type = element_at;
    using size_type = std::size_t;
    using pointer = element_at*;
    using const_pointer = element_at const*;
    template <typename other_element_at> struct rebind {
        using other = aligned_allocator_gt<other_element_at>;
    };

    constexpr std::size_t alignment() const { return alignment_ak; }

    pointer allocate(size_type length) const {
        std::size_t length_bytes = alignment_ak * divide_round_up<alignment_ak>(length * sizeof(value_type));
        std::size_t alignment = alignment_ak;
        // void* result = nullptr;
        // int status = posix_memalign(&result, alignment, length_bytes);
        // return status == 0 ? (pointer)result : nullptr;
#if defined(USEARCH_DEFINED_WINDOWS)
        return (pointer)_aligned_malloc(length_bytes, alignment);
#else
        return (pointer)aligned_alloc(alignment, length_bytes);
#endif
    }

    void deallocate(pointer begin, size_type) const {
#if defined(USEARCH_DEFINED_WINDOWS)
        _aligned_free(begin);
#else
        free(begin);
#endif
    }
};

using aligned_allocator_t = aligned_allocator_gt<>;

class page_allocator_t {
  public:
    static constexpr std::size_t page_size() { return 4096; }

    /**
     *  @brief Allocates an @b uninitialized block of memory of the specified size.
     *  @param count_bytes The number of bytes to allocate.
     *  @return A pointer to the allocated memory block, or `nullptr` if allocation fails.
     */
    byte_t* allocate(std::size_t count_bytes) const noexcept {
        count_bytes = divide_round_up(count_bytes, page_size()) * page_size();
#if defined(USEARCH_DEFINED_WINDOWS)
        return (byte_t*)(::VirtualAlloc(NULL, count_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
#else
        return (byte_t*)mmap(NULL, count_bytes, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
#endif
    }

    void deallocate(byte_t* page_pointer, std::size_t count_bytes) const noexcept {
#if defined(USEARCH_DEFINED_WINDOWS)
        ::VirtualFree(page_pointer, 0, MEM_RELEASE);
#else
        count_bytes = divide_round_up(count_bytes, page_size()) * page_size();
        munmap(page_pointer, count_bytes);
#endif
    }
};

/**
 *  @brief  Memory-mapping allocator designed for "alloc many, free at once" usage patterns.
 *          @b Thread-safe, @b except constructors and destructors.
 *
 *  Using this memory allocator won't affect your overall speed much, as that is not the bottleneck.
 *  However, it can drastically improve memory usage especcially for huge indexes of small vectors.
 */
template <std::size_t alignment_ak = 1> class memory_mapping_allocator_gt {

    static constexpr std::size_t min_capacity() { return 1024 * 1024 * 4; }
    static constexpr std::size_t capacity_multiplier() { return 2; }
    static constexpr std::size_t head_size() {
        /// Pointer to the the previous arena and the size of the current one.
        return divide_round_up<alignment_ak>(sizeof(byte_t*) + sizeof(std::size_t)) * alignment_ak;
    }

    std::mutex mutex_;
    byte_t* last_arena_ = nullptr;
    std::size_t last_usage_ = head_size();
    std::size_t last_capacity_ = min_capacity();
    std::size_t wasted_space_ = 0;

  public:
    using value_type = byte_t;
    using size_type = std::size_t;
    using pointer = byte_t*;
    using const_pointer = byte_t const*;

    memory_mapping_allocator_gt() = default;
    memory_mapping_allocator_gt(memory_mapping_allocator_gt&& other) noexcept
        : last_arena_(exchange(other.last_arena_, nullptr)), last_usage_(exchange(other.last_usage_, 0)),
          last_capacity_(exchange(other.last_capacity_, 0)), wasted_space_(exchange(other.wasted_space_, 0)) {}

    memory_mapping_allocator_gt& operator=(memory_mapping_allocator_gt&& other) noexcept {
        std::swap(last_arena_, other.last_arena_);
        std::swap(last_usage_, other.last_usage_);
        std::swap(last_capacity_, other.last_capacity_);
        std::swap(wasted_space_, other.wasted_space_);
        return *this;
    }

    ~memory_mapping_allocator_gt() noexcept { reset(); }

    /**
     *  @brief Discards all previously allocated memory buffers.
     */
    void reset() noexcept {
        byte_t* last_arena = last_arena_;
        while (last_arena) {
            byte_t* previous_arena = nullptr;
            std::memcpy(&previous_arena, last_arena, sizeof(byte_t*));
            std::size_t last_cap = 0;
            std::memcpy(&last_cap, last_arena + sizeof(byte_t*), sizeof(std::size_t));
            page_allocator_t{}.deallocate(last_arena, last_cap);
            last_arena = previous_arena;
        }

        // Clear the references:
        last_arena_ = nullptr;
        last_usage_ = head_size();
        last_capacity_ = min_capacity();
        wasted_space_ = 0;
    }

    /**
     *  @brief Copy constructor.
     *  @note This is a no-op copy constructor since the allocator is not copyable.
     */
    memory_mapping_allocator_gt(memory_mapping_allocator_gt const&) noexcept {}

    /**
     *  @brief Copy assignment operator.
     *  @note This is a no-op copy assignment operator since the allocator is not copyable.
     *  @return Reference to the allocator after the assignment.
     */
    memory_mapping_allocator_gt& operator=(memory_mapping_allocator_gt const&) noexcept {
        reset();
        return *this;
    }

    /**
     *  @brief Allocates an @b uninitialized block of memory of the specified size.
     *  @param count_bytes The number of bytes to allocate.
     *  @return A pointer to the allocated memory block, or `nullptr` if allocation fails.
     */
    inline byte_t* allocate(std::size_t count_bytes) noexcept {
        std::size_t extended_bytes = divide_round_up<alignment_ak>(count_bytes) * alignment_ak;
        std::unique_lock<std::mutex> lock(mutex_);
        if (!last_arena_ || (last_usage_ + extended_bytes >= last_capacity_)) {
            std::size_t new_cap = (std::max)(last_capacity_, ceil2(extended_bytes)) * capacity_multiplier();
            byte_t* new_arena = page_allocator_t{}.allocate(new_cap);
            if (!new_arena)
                return nullptr;
            std::memcpy(new_arena, &last_arena_, sizeof(byte_t*));
            std::memcpy(new_arena + sizeof(byte_t*), &new_cap, sizeof(std::size_t));

            wasted_space_ += total_reserved();
            last_arena_ = new_arena;
            last_capacity_ = new_cap;
            last_usage_ = head_size();
        }

        wasted_space_ += extended_bytes - count_bytes;
        return last_arena_ + exchange(last_usage_, last_usage_ + extended_bytes);
    }

    /**
     *  @brief Returns the amount of memory used by the allocator across all arenas.
     *  @return The amount of space in bytes.
     */
    std::size_t total_allocated() const noexcept {
        if (!last_arena_)
            return 0;
        std::size_t total_used = 0;
        std::size_t last_capacity = last_capacity_;
        do {
            total_used += last_capacity;
            last_capacity /= capacity_multiplier();
        } while (last_capacity >= min_capacity());
        return total_used;
    }

    /**
     *  @brief Returns the amount of wasted space due to alignment.
     *  @return The amount of wasted space in bytes.
     */
    std::size_t total_wasted() const noexcept { return wasted_space_; }

    /**
     *  @brief Returns the amount of remaining memory already reserved but not yet used.
     *  @return The amount of reserved memory in bytes.
     */
    std::size_t total_reserved() const noexcept { return last_arena_ ? last_capacity_ - last_usage_ : 0; }

    /**
     *  @warning The very first memory de-allocation discards all the arenas!
     */
    void deallocate(byte_t* = nullptr, std::size_t = 0) noexcept { reset(); }
};

using memory_mapping_allocator_t = memory_mapping_allocator_gt<>;

/**
 *  @brief  C++11 userspace implementation of an oversimplified `std::shared_mutex`,
 *          that assumes rare interleaving of shared and unique locks. It's not fair,
 *          but requires only a single 32-bit atomic integer to work.
 */
class unfair_shared_mutex_t {
    /** Any positive integer describes the number of concurrent readers */
    enum state_t : std::int32_t {
        idle_k = 0,
        writing_k = -1,
    };
    std::atomic<std::int32_t> state_{idle_k};

  public:
    inline void lock() noexcept {
        std::int32_t raw;
    relock:
        raw = idle_k;
        if (!state_.compare_exchange_weak(raw, writing_k, std::memory_order_acquire, std::memory_order_relaxed)) {
            std::this_thread::yield();
            goto relock;
        }
    }

    inline void unlock() noexcept { state_.store(idle_k, std::memory_order_release); }

    inline void lock_shared() noexcept {
        std::int32_t raw;
    relock_shared:
        raw = state_.load(std::memory_order_acquire);
        // Spin while it's uniquely locked
        if (raw == writing_k) {
            std::this_thread::yield();
            goto relock_shared;
        }
        // Try incrementing the counter
        if (!state_.compare_exchange_weak(raw, raw + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            std::this_thread::yield();
            goto relock_shared;
        }
    }

    inline void unlock_shared() noexcept { state_.fetch_sub(1, std::memory_order_release); }

    /**
     *  @brief Try upgrades the current `lock_shared()` to a unique `lock()` state.
     */
    inline bool try_escalate() noexcept {
        std::int32_t one_read = 1;
        return state_.compare_exchange_weak(one_read, writing_k, std::memory_order_acquire, std::memory_order_relaxed);
    }

    /**
     *  @brief Escalates current lock potentially loosing control in the middle.
     *  It's a shortcut for `try_escalate`-`unlock_shared`-`lock` trio.
     */
    inline void unsafe_escalate() noexcept {
        if (!try_escalate()) {
            unlock_shared();
            lock();
        }
    }

    /**
     *  @brief Upgrades the current `lock_shared()` to a unique `lock()` state.
     */
    inline void escalate() noexcept {
        while (!try_escalate())
            std::this_thread::yield();
    }

    /**
     *  @brief De-escalation of a previously escalated state.
     */
    inline void de_escalate() noexcept {
        std::int32_t one_read = 1;
        state_.store(one_read, std::memory_order_release);
    }
};

template <typename mutex_at = unfair_shared_mutex_t> class shared_lock_gt {
    mutex_at& mutex_;

  public:
    inline explicit shared_lock_gt(mutex_at& m) noexcept : mutex_(m) { mutex_.lock_shared(); }
    inline ~shared_lock_gt() noexcept { mutex_.unlock_shared(); }
};

/**
 *  @brief  Utility class used to cast arrays of one scalar type to another,
 *          avoiding unnecessary conversions.
 */
template <typename from_scalar_at, typename to_scalar_at> struct cast_gt {
    inline bool operator()(byte_t const* input, std::size_t dim, byte_t* output) const {
        from_scalar_at const* typed_input = reinterpret_cast<from_scalar_at const*>(input);
        to_scalar_at* typed_output = reinterpret_cast<to_scalar_at*>(output);
        auto converter = [](from_scalar_at from) { return to_scalar_at(from); };
        std::transform(typed_input, typed_input + dim, typed_output, converter);
        return true;
    }
};

template <> struct cast_gt<f32_t, f32_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <> struct cast_gt<f64_t, f64_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <> struct cast_gt<f16_bits_t, f16_bits_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <> struct cast_gt<i8_t, i8_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <> struct cast_gt<b1x8_t, b1x8_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <typename from_scalar_at> struct cast_gt<from_scalar_at, b1x8_t> {
    inline bool operator()(byte_t const* input, std::size_t dim, byte_t* output) const {
        from_scalar_at const* typed_input = reinterpret_cast<from_scalar_at const*>(input);
        unsigned char* typed_output = reinterpret_cast<unsigned char*>(output);
        for (std::size_t i = 0; i != dim; ++i)
            typed_output[i / CHAR_BIT] |= bool(typed_input[i]) ? (128 >> (i & (CHAR_BIT - 1))) : 0;
        return true;
    }
};

template <typename to_scalar_at> struct cast_gt<b1x8_t, to_scalar_at> {
    inline bool operator()(byte_t const* input, std::size_t dim, byte_t* output) const {
        unsigned char const* typed_input = reinterpret_cast<unsigned char const*>(input);
        to_scalar_at* typed_output = reinterpret_cast<to_scalar_at*>(output);
        for (std::size_t i = 0; i != dim; ++i)
            typed_output[i] = bool(typed_input[i / CHAR_BIT] & (128 >> (i & (CHAR_BIT - 1))));
        return true;
    }
};

/**
 *  @brief  Numeric type for uniformly-distributed floating point
 *          values within [-1,1] range, quantized to integers [-100,100].
 */
class i8_converted_t {
    std::int8_t int8_{};

  public:
    constexpr static float divisor_k = 100.f;
    constexpr static std::int8_t min_k = -100;
    constexpr static std::int8_t max_k = 100;

    inline i8_converted_t() noexcept : int8_(0) {}
    inline i8_converted_t(bool v) noexcept : int8_(v ? max_k : 0) {}

    inline i8_converted_t(i8_converted_t&&) = default;
    inline i8_converted_t& operator=(i8_converted_t&&) = default;
    inline i8_converted_t(i8_converted_t const&) = default;
    inline i8_converted_t& operator=(i8_converted_t const&) = default;

    inline operator float() const noexcept { return float(int8_) / divisor_k; }
    inline operator f16_t() const noexcept { return float(int8_) / divisor_k; }
    inline operator double() const noexcept { return double(int8_) / divisor_k; }
    inline explicit operator bool() const noexcept { return int8_ > (max_k / 2); }
    inline explicit operator std::int8_t() const noexcept { return int8_; }
    inline explicit operator std::int16_t() const noexcept { return int8_; }
    inline explicit operator std::int32_t() const noexcept { return int8_; }
    inline explicit operator std::int64_t() const noexcept { return int8_; }

    inline i8_converted_t(f16_t v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
    inline i8_converted_t(float v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
    inline i8_converted_t(double v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
};

f16_bits_t::f16_bits_t(i8_converted_t v) noexcept : uint16_(f32_to_f16(v)) {}

template <> struct cast_gt<i8_t, f16_t> : public cast_gt<i8_converted_t, f16_t> {};
template <> struct cast_gt<i8_t, f32_t> : public cast_gt<i8_converted_t, f32_t> {};
template <> struct cast_gt<i8_t, f64_t> : public cast_gt<i8_converted_t, f64_t> {};

template <> struct cast_gt<f16_t, i8_t> : public cast_gt<f16_t, i8_converted_t> {};
template <> struct cast_gt<f32_t, i8_t> : public cast_gt<f32_t, i8_converted_t> {};
template <> struct cast_gt<f64_t, i8_t> : public cast_gt<f64_t, i8_converted_t> {};

/**
 *  @brief  Inner (Dot) Product distance.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct metric_ip_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t dim) const noexcept {
        result_t ab{};
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
 *          is recommended over `::metric_ip_gt` for low-precision scalars.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct metric_cos_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;

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

        result_t result_if_zero[2][2];
        result_if_zero[0][0] = 1 - ab / (std::sqrt(a2) * std::sqrt(b2));
        result_if_zero[0][1] = result_if_zero[1][0] = 1;
        result_if_zero[1][1] = 0;
        return result_if_zero[a2 == 0][b2 == 0];
    }
};

/**
 *  @brief  Squared Euclidean (L2) distance.
 *          Square root is avoided at the end, as it won't affect the ordering.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct metric_l2sq_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;

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
template <typename scalar_at = std::uint64_t, typename result_at = std::size_t> struct metric_hamming_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;
    static_assert( //
        std::is_unsigned<scalar_t>::value ||
            (std::is_enum<scalar_t>::value && std::is_unsigned<typename std::underlying_type<scalar_t>::type>::value),
        "Hamming distance requires unsigned integral words");

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
template <typename scalar_at = std::uint64_t, typename result_at = float> struct metric_tanimoto_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;
    static_assert( //
        std::is_unsigned<scalar_t>::value ||
            (std::is_enum<scalar_t>::value && std::is_unsigned<typename std::underlying_type<scalar_t>::type>::value),
        "Tanimoto distance requires unsigned integral words");
    static_assert(std::is_floating_point<result_t>::value, "Tanimoto distance will be a fraction");

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
template <typename scalar_at = std::uint64_t, typename result_at = float> struct metric_sorensen_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;
    static_assert( //
        std::is_unsigned<scalar_t>::value ||
            (std::is_enum<scalar_t>::value && std::is_unsigned<typename std::underlying_type<scalar_t>::type>::value),
        "Sorensen-Dice distance requires unsigned integral words");
    static_assert(std::is_floating_point<result_t>::value, "Sorensen-Dice distance will be a fraction");

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
 *          Similar to `metric_tanimoto_gt` for dense representations.
 */
template <typename scalar_at = std::int32_t, typename result_at = float> struct metric_jaccard_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;
    static_assert(!std::is_floating_point<scalar_t>::value, "Jaccard distance requires integral scalars");

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
 *  @brief  Measures Pearson Correlation between two sequences.
 */
template <typename scalar_at = float, typename result_at = float> struct metric_pearson_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;

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

struct cos_i8_t {
    using scalar_t = i8_t;
    using result_t = f32_t;

    inline result_t operator()(i8_t const* a, i8_t const* b, std::size_t dim) const noexcept {
        std::int32_t ab{}, a2{}, b2{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab, a2, b2)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dim; i++) {
            std::int16_t ai{a[i]};
            std::int16_t bi{b[i]};
            ab += ai * bi;
            a2 += square(ai);
            b2 += square(bi);
        }
        return (ab != 0) ? (1.f - ab / (std::sqrt(a2) * std::sqrt(b2))) : 0;
    }
};

struct l2sq_i8_t {
    using scalar_t = i8_t;
    using result_t = f32_t;

    inline result_t operator()(i8_t const* a, i8_t const* b, std::size_t dim) const noexcept {
        std::int32_t ab_deltas_sq{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab_deltas_sq)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dim; i++)
            ab_deltas_sq += square(std::int16_t(a[i]) - std::int16_t(b[i]));
        return ab_deltas_sq;
    }
};

/**
 *  @brief  Haversine distance for the shortest distance between two nodes on
 *          the surface of a 3D sphere, defined with latitude and longitude.
 */
template <typename scalar_at = float, typename result_at = scalar_at> struct metric_haversine_gt {
    using scalar_t = scalar_at;
    using result_t = result_at;
    static_assert(!std::is_integral<scalar_t>::value, "Latitude and longitude must be floating-node");

    inline result_t operator()(scalar_t const* a, scalar_t const* b, std::size_t = 2) const noexcept {
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

using distance_punned_t = float;
using span_punned_t = span_gt<byte_t const>;

/**
 *  @brief  Type-punned metric class, that can wrap an existing STL `std::function`.
 *          Additional annotation is useful for
 */
class metric_punned_t {
  public:
    using scalar_t = byte_t;
    using result_t = distance_punned_t;
    using stl_function_t = std::function<float(byte_t const*, byte_t const*)>;

  private:
    stl_function_t stl_function_;
    std::size_t dimensions_ = 0;
    metric_kind_t metric_kind_ = metric_kind_t::unknown_k;
    scalar_kind_t scalar_kind_ = scalar_kind_t::unknown_k;
    isa_kind_t isa_kind_ = isa_kind_t::auto_k;

  public:
    /**
     *  @brief  Computes the distance between two vectors of fixed length.
     *  ! The only relevant function in the object. Everything else is just dynamic dispatch.
     */
    inline result_t operator()(byte_t const* a, byte_t const* b) const noexcept { return stl_function_(a, b); }

    inline metric_punned_t() = default;
    inline metric_punned_t(metric_punned_t const&) = default;
    inline metric_punned_t& operator=(metric_punned_t const&) = default;

    inline metric_punned_t( //
        stl_function_t stl_function, std::size_t dimensions = 0, metric_kind_t metric_kind = metric_kind_t::unknown_k,
        scalar_kind_t scalar_kind = scalar_kind_t::unknown_k, isa_kind_t isa_kind = isa_kind_t::auto_k)
        : stl_function_(stl_function), dimensions_(dimensions), metric_kind_(metric_kind), scalar_kind_(scalar_kind),
          isa_kind_(isa_kind) {}

    inline metric_punned_t( //
        std::size_t dimensions, metric_kind_t metric_kind = metric_kind_t::cos_k,
        scalar_kind_t scalar_kind = scalar_kind_t::f32_k) {
        std::size_t bytes_per_vector = divide_round_up<CHAR_BIT>(dimensions * bits_per_scalar(scalar_kind));
        *this = make_(bytes_per_vector, metric_kind, scalar_kind);
        dimensions_ = dimensions;
    }

    std::size_t dimensions() const noexcept { return dimensions_; }
    metric_kind_t metric_kind() const noexcept { return metric_kind_; }
    scalar_kind_t scalar_kind() const noexcept { return scalar_kind_; }
    isa_kind_t isa_kind() const noexcept { return isa_kind_; }

    inline std::size_t bytes_per_vector() const noexcept {
        return divide_round_up<CHAR_BIT>(dimensions_ * bits_per_scalar(scalar_kind_));
    }

    inline std::size_t scalar_words() const noexcept {
        return divide_round_up(dimensions_ * bits_per_scalar(scalar_kind_), bits_per_scalar_word(scalar_kind_));
    }

  private:
    static metric_punned_t make_(std::size_t bytes_per_vector, metric_kind_t metric_kind, scalar_kind_t scalar_kind) {

        switch (metric_kind) {
        case metric_kind_t::ip_k: return ip_metric_(bytes_per_vector, scalar_kind);
        case metric_kind_t::cos_k: return cos_metric_(bytes_per_vector, scalar_kind);
        case metric_kind_t::l2sq_k: return l2sq_metric_(bytes_per_vector, scalar_kind);
        case metric_kind_t::pearson_k: return pearson_metric_(bytes_per_vector, scalar_kind);
        case metric_kind_t::haversine_k: return haversine_metric_(scalar_kind);

        case metric_kind_t::hamming_k:
            return {to_stl_<metric_hamming_gt<b1x8_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::hamming_k,
                    scalar_kind_t::b1x8_k, isa_kind_t::auto_k};

        case metric_kind_t::jaccard_k: // Equivalent to Tanimoto
        case metric_kind_t::tanimoto_k:
            return {to_stl_<metric_tanimoto_gt<b1x8_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::tanimoto_k,
                    scalar_kind_t::b1x8_k, isa_kind_t::auto_k};

        case metric_kind_t::sorensen_k:
            return {to_stl_<metric_sorensen_gt<b1x8_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::sorensen_k,
                    scalar_kind_t::b1x8_k, isa_kind_t::auto_k};

        default: return {};
        }
    }

    template <typename typed_at> static stl_function_t to_stl_(std::size_t bytes) {
        using scalar_t = typename typed_at::scalar_t;
        return [=](byte_t const* a, byte_t const* b) -> result_t {
            return typed_at{}((scalar_t const*)a, (scalar_t const*)b, bytes / sizeof(scalar_t));
        };
    }

    template <typename scalar_at>
    static stl_function_t pun_stl_(std::function<result_t(scalar_at const*, scalar_at const*)> typed) {
        return [=](byte_t const* a, byte_t const* b) -> result_t {
            return typed((scalar_at const*)a, (scalar_at const*)b);
        };
    }

    // clang-format off
    static metric_punned_t ip_metric_f32_(std::size_t bytes_per_vector) {
        #if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_kind_t::sve_k)) return {pun_stl_<f32_t>([=](f32_t const* a, f32_t const* b) { return simsimd_dot_f32_sve(a, b, bytes_per_vector / 4); }), bytes_per_vector, metric_kind_t::ip_k, scalar_kind_t::f32_k, isa_kind_t::sve_k};
        if (hardware_supports(isa_kind_t::neon_k) && bytes_per_vector % 16 == 0) return {pun_stl_<f32_t>([=](f32_t const* a, f32_t const* b) { return simsimd_dot_f32x4_neon(a, b, bytes_per_vector / 4); }), bytes_per_vector, metric_kind_t::ip_k, scalar_kind_t::f32_k, isa_kind_t::neon_k};
        if (hardware_supports(isa_kind_t::avx2_k) && bytes_per_vector % 16 == 0) return {pun_stl_<f32_t>([=](f32_t const* a, f32_t const* b) { return simsimd_dot_f32x4_avx2(a, b, bytes_per_vector / 4); }), bytes_per_vector, metric_kind_t::ip_k, scalar_kind_t::f32_k, isa_kind_t::avx2_k};
        #endif
        return {to_stl_<metric_ip_gt<f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::ip_k, scalar_kind_t::f32_k, isa_kind_t::auto_k};
    }

    static metric_punned_t cos_metric_f16_(std::size_t bytes_per_vector) {
        #if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_kind_t::avx512_k) && bytes_per_vector % 32 == 0) return {pun_stl_<simsimd_f16_t>([=](simsimd_f16_t const* a, simsimd_f16_t const* b) { return simsimd_cos_f16x16_avx512(a, b, bytes_per_vector / 2); }), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::f16_k, isa_kind_t::avx512_k};
        if (hardware_supports(isa_kind_t::neon_k) && bytes_per_vector % 8 == 0) return {pun_stl_<simsimd_f16_t>([=](simsimd_f16_t const* a, simsimd_f16_t const* b) { return simsimd_cos_f16x4_neon(a, b, bytes_per_vector / 2); }), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::f16_k, isa_kind_t::neon_k};
        #endif
        return {to_stl_<metric_cos_gt<f16_t, f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::f16_k, isa_kind_t::auto_k};
    }

    static metric_punned_t cos_metric_i8_(std::size_t bytes_per_vector) {
        #if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_kind_t::neon_k) && bytes_per_vector % 16 == 0) return {pun_stl_<int8_t>([=](int8_t const* a, int8_t const* b) { return simsimd_cos_i8x16_neon(a, b, bytes_per_vector); }), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::i8_k, isa_kind_t::neon_k};
        #endif
        return {to_stl_<cos_i8_t>(bytes_per_vector), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::i8_k, isa_kind_t::auto_k};
    }

    static metric_punned_t ip_metric_(std::size_t bytes_per_vector, scalar_kind_t scalar_kind) {        
        switch (scalar_kind) { // The two most common numeric types for the most common metric have optimized versions
        case scalar_kind_t::f32_k: return ip_metric_f32_(bytes_per_vector);
        case scalar_kind_t::f16_k: return cos_metric_f16_(bytes_per_vector); // Dot-product accumulates error, Cosine-distance normalizes it
        case scalar_kind_t::i8_k:  return cos_metric_i8_(bytes_per_vector);
        case scalar_kind_t::f64_k: return {to_stl_<metric_ip_gt<f64_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::ip_k, scalar_kind_t::f64_k, isa_kind_t::auto_k};
        default: return {};
        }
    }

    static metric_punned_t l2sq_metric_(std::size_t bytes_per_vector, scalar_kind_t scalar_kind) {
        switch (scalar_kind) {
        case scalar_kind_t::i8_k: return {to_stl_<l2sq_i8_t>(bytes_per_vector), bytes_per_vector, metric_kind_t::l2sq_k, scalar_kind_t::i8_k, isa_kind_t::auto_k};
        case scalar_kind_t::f16_k: return {to_stl_<metric_l2sq_gt<f16_t, f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::l2sq_k, scalar_kind_t::f16_k, isa_kind_t::auto_k};
        case scalar_kind_t::f32_k: return {to_stl_<metric_l2sq_gt<f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::l2sq_k, scalar_kind_t::f32_k, isa_kind_t::auto_k};
        case scalar_kind_t::f64_k: return {to_stl_<metric_l2sq_gt<f64_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::l2sq_k, scalar_kind_t::f64_k, isa_kind_t::auto_k};
        default: return {};
        }
    }

    static metric_punned_t cos_metric_(std::size_t bytes_per_vector, scalar_kind_t scalar_kind) {
        switch (scalar_kind) {
        case scalar_kind_t::i8_k: return cos_metric_i8_(bytes_per_vector);
        case scalar_kind_t::f16_k: return cos_metric_f16_(bytes_per_vector);
        case scalar_kind_t::f32_k: return {to_stl_<metric_cos_gt<f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::f32_k, isa_kind_t::auto_k};
        case scalar_kind_t::f64_k: return {to_stl_<metric_cos_gt<f64_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::cos_k, scalar_kind_t::f64_k, isa_kind_t::auto_k};
        default: return {};
        }
    }

    static metric_punned_t haversine_metric_(scalar_kind_t scalar_kind) {
        std::size_t bytes_per_vector = 2u * bits_per_scalar(scalar_kind) / CHAR_BIT;
        switch (scalar_kind) {
        case scalar_kind_t::f16_k: return {to_stl_<metric_haversine_gt<f16_t, f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::haversine_k, scalar_kind_t::f16_k, isa_kind_t::auto_k};
        case scalar_kind_t::f32_k: return {to_stl_<metric_haversine_gt<f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::haversine_k, scalar_kind_t::f32_k, isa_kind_t::auto_k};
        case scalar_kind_t::f64_k: return {to_stl_<metric_haversine_gt<f64_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::haversine_k, scalar_kind_t::f64_k, isa_kind_t::auto_k};
        default: return {};
        }
    }

    static metric_punned_t pearson_metric_(std::size_t bytes_per_vector, scalar_kind_t scalar_kind) {
        switch (scalar_kind) {
        case scalar_kind_t::i8_k: return {to_stl_<metric_pearson_gt<i8_t, f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::pearson_k, scalar_kind_t::i8_k, isa_kind_t::auto_k};
        case scalar_kind_t::f16_k: return {to_stl_<metric_pearson_gt<f16_t, f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::pearson_k, scalar_kind_t::f16_k, isa_kind_t::auto_k};
        case scalar_kind_t::f32_k: return {to_stl_<metric_pearson_gt<f32_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::pearson_k, scalar_kind_t::f32_k, isa_kind_t::auto_k};
        case scalar_kind_t::f64_k: return {to_stl_<metric_pearson_gt<f64_t>>(bytes_per_vector), bytes_per_vector, metric_kind_t::pearson_k, scalar_kind_t::f64_k, isa_kind_t::auto_k};
        default: return {};
        }
    }
    // clang-format on
};

} // namespace usearch
} // namespace unum
