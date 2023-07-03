#pragma once
#include <stdlib.h> // `aligned_alloc`

#include <cstring> // `std::strncmp`
#include <numeric> // `std::iota`
#include <thread>  // `std::thread`
#include <vector>  // `std::vector`

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
#define USEARCH_USE_NATIVE_F16 0
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

enum class isa_t {
    auto_k,
    neon_k,
    sve_k,
    avx2_k,
    avx512_k,
};

inline char const* isa_name(isa_t isa) noexcept {
    switch (isa) {
    case isa_t::auto_k: return "auto";
    case isa_t::neon_k: return "neon";
    case isa_t::sve_k: return "sve";
    case isa_t::avx2_k: return "avx2";
    case isa_t::avx512_k: return "avx512";
    default: return "";
    }
}

inline bool hardware_supports(isa_t isa) noexcept {
#if defined(USEARCH_DEFINED_ARM) && defined(USEARCH_DEFINED_LINUX)
    unsigned long capabilities = getauxval(AT_HWCAP);
    switch (isa) {
    case isa_t::neon_k: return true; // Must be supported on 64-bit Arm
    case isa_t::sve_k: return capabilities & HWCAP_SVE;
    default: return false;
    }
#endif

#if defined(USEARCH_DEFINED_X86) && defined(USEARCH_DEFINED_GCC)
    __builtin_cpu_init();
    switch (isa) {
    case isa_t::avx2_k: return __builtin_cpu_supports("avx2");
    case isa_t::avx512_k: return __builtin_cpu_supports("avx512f");
    default: return false;
    }
#endif

    (void)isa;
    return false;
}

inline std::size_t bytes_per_scalar(scalar_kind_t accuracy) noexcept {
    switch (accuracy) {
    case scalar_kind_t::f32_k: return 4;
    case scalar_kind_t::f16_k: return 2;
    case scalar_kind_t::f64_k: return 8;
    case scalar_kind_t::f8_k: return 1;
    case scalar_kind_t::b1x8_k: return 1;
    default: return 0;
    }
}

inline char const* scalar_kind_name(scalar_kind_t accuracy) noexcept {
    switch (accuracy) {
    case scalar_kind_t::f32_k: return "f32";
    case scalar_kind_t::f16_k: return "f16";
    case scalar_kind_t::f64_k: return "f64";
    case scalar_kind_t::f8_k: return "f8";
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
    else if (str_equals(name, len, "f8"))
        parsed.result = scalar_kind_t::f8_k;
    else
        parsed.failed("Unknown type, choose: f32, f16, f64, f8");
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

using punned_distance_t = float;
using punned_vector_view_t = span_gt<byte_t const>;

class f8_bits_t;
class f16_bits_t;

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

    inline f16_bits_t(f8_bits_t) noexcept;
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
 *  @brief  Numeric type for uniformly-distributed floating point
 *          values within [-1,1] range, quantized to integers [-100,100].
 */
class f8_bits_t {
    std::int8_t int8_{};

  public:
    constexpr static float divisor_k = 100.f;
    constexpr static std::int8_t min_k = -100;
    constexpr static std::int8_t max_k = 100;

    inline f8_bits_t() noexcept : int8_(0) {}
    inline f8_bits_t(bool v) noexcept : int8_(v ? max_k : 0) {}

    inline f8_bits_t(f8_bits_t&&) = default;
    inline f8_bits_t& operator=(f8_bits_t&&) = default;
    inline f8_bits_t(f8_bits_t const&) = default;
    inline f8_bits_t& operator=(f8_bits_t const&) = default;

    inline operator float() const noexcept { return float(int8_) / divisor_k; }
    inline operator f16_t() const noexcept { return float(int8_) / divisor_k; }
    inline operator double() const noexcept { return double(int8_) / divisor_k; }
    inline explicit operator bool() const noexcept { return int8_ > (max_k / 2); }
    inline explicit operator std::int8_t() const noexcept { return int8_; }
    inline explicit operator std::int16_t() const noexcept { return int8_; }
    inline explicit operator std::int32_t() const noexcept { return int8_; }
    inline explicit operator std::int64_t() const noexcept { return int8_; }

    inline f8_bits_t(f16_t v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
    inline f8_bits_t(float v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
    inline f8_bits_t(double v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
};

inline f16_bits_t::f16_bits_t(f8_bits_t v) noexcept : f16_bits_t(float(v)) {}

struct uuid_t {
    std::uint8_t octets[16];
};

class executor_stl_t {
    std::size_t threads_count_{};

  public:
    executor_stl_t(std::size_t threads_count) noexcept
        : threads_count_(threads_count ? threads_count : std::thread::hardware_concurrency()) {}

    template <typename thread_aware_function_at>
    void execute_bulk(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept(false) {
        std::vector<std::thread> threads_pool;
        std::size_t tasks_per_thread = (tasks / threads_count_) + ((tasks % threads_count_) != 0);
        for (std::size_t thread_idx = 0; thread_idx != threads_count_; ++thread_idx) {
            threads_pool.emplace_back([=]() {
                for (std::size_t task_idx = thread_idx * tasks_per_thread;
                     task_idx < (std::min)(tasks, thread_idx * tasks_per_thread + tasks_per_thread); ++task_idx)
                    thread_aware_function(thread_idx, task_idx);
            });
        }
        for (std::size_t thread_idx = 0; thread_idx != threads_count_; ++thread_idx)
            threads_pool[thread_idx].join();
    }
};

#if USEARCH_USE_OPENMP

class executor_openmp_t {
  public:
    executor_openmp_t(std::size_t threads_count) noexcept {
        omp_set_num_threads(threads_count ? threads_count : std::thread::hardware_concurrency());
    }

    template <typename thread_aware_function_at>
    void execute_bulk(std::size_t tasks, thread_aware_function_at&& thread_aware_function) noexcept(false) {
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < tasks; ++i)
            thread_aware_function(omp_get_thread_num(), i);
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

#if !defined(USEARCH_DEFINED_WINDOWS)

/**
 *  @brief  Memory-mapping allocator designed for "alloc many, free at once" usage patterns.
 *          Thread-safe.
 *
 *  Using this memory allocator won't affect your overall speed much, as that is not the bottleneck.
 *  However, it can drastically improve memory usage especcially for huge indexes of small vectors.
 */
template <std::size_t alignment_ak = 1> class memory_mapping_allocator_gt {

    static constexpr std::size_t min_size() { return 1024 * 1024 * 4; }
    static constexpr std::size_t head_size() {
        /// Pointer to the the previous arena and the size of the current one.
        return divide_round_up<alignment_ak>(sizeof(byte_t*) + sizeof(std::size_t)) * alignment_ak;
    }

    std::mutex mutex_;
    byte_t* last_arena_ = nullptr;
    std::size_t last_usage_ = head_size();
    std::size_t last_capacity_ = min_size();

    void reset() noexcept {
        byte_t* last_arena = last_arena_;
        while (last_arena) {
            byte_t* previous_arena;
            std::memcpy(&previous_arena, last_arena, sizeof(byte_t*));
            std::size_t current_size;
            std::memcpy(&current_size, last_arena + sizeof(byte_t*), sizeof(std::size_t));
            munmap(last_arena, current_size);
            last_arena = previous_arena;
        }

        // Clear the references:
        last_arena_ = nullptr;
        last_usage_ = head_size();
        last_capacity_ = min_size();
    }

  public:
    using value_type = byte_t;
    using size_type = std::size_t;
    using pointer = byte_t*;
    using const_pointer = byte_t const*;

    memory_mapping_allocator_gt() = default;
    memory_mapping_allocator_gt(memory_mapping_allocator_gt&& other) noexcept
        : last_arena_(other.last_arena_), last_usage_(other.last_usage_), last_capacity_(other.last_capacity_) {}
    memory_mapping_allocator_gt& operator=(memory_mapping_allocator_gt&& other) noexcept {
        std::swap(last_arena_, other.last_arena_);
        std::swap(last_usage_, other.last_usage_);
        std::swap(last_capacity_, other.last_capacity_);
        return *this;
    }

    ~memory_mapping_allocator_gt() noexcept { reset(); }

    inline byte_t* allocate(std::size_t count_bytes) noexcept {
        count_bytes = divide_round_up<alignment_ak>(count_bytes) * alignment_ak;

        std::unique_lock<std::mutex> lock(mutex_);
        if (!last_arena_ || (last_usage_ + count_bytes > last_capacity_)) {
            std::size_t new_capacity = last_capacity_ * 2;
            int prot = PROT_WRITE | PROT_READ;
            int flags = MAP_PRIVATE | MAP_ANONYMOUS;
            byte_t* new_arena = (byte_t*)mmap(NULL, new_capacity, prot, flags, 0, 0);
            std::memcpy(new_arena, &last_arena_, sizeof(byte_t*));
            std::memcpy(new_arena + sizeof(byte_t*), &new_capacity, sizeof(std::size_t));

            last_arena_ = new_arena;
            last_capacity_ = new_capacity;
            last_usage_ = head_size();
        }

        return last_arena_ + exchange(last_usage_, last_usage_ + count_bytes);
    }

    /**
     *  @warning The very first memory de-allocation discards all the arenas!
     */
    void deallocate(byte_t*, std::size_t) noexcept { reset(); }
};

using memory_mapping_allocator_t = memory_mapping_allocator_gt<>;

#else

using memory_mapping_allocator_t = aligned_allocator_t;

#endif

template <typename from_scalar_at, typename to_scalar_at> struct cast_gt {
    inline bool operator()(byte_t const* input, std::size_t dimensions, byte_t* output) const {
        from_scalar_at const* typed_input = reinterpret_cast<from_scalar_at const*>(input);
        to_scalar_at* typed_output = reinterpret_cast<to_scalar_at*>(output);
        auto converter = [](from_scalar_at from) { return to_scalar_at(from); };
        std::transform(typed_input, typed_input + dimensions, typed_output, converter);
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

template <> struct cast_gt<f8_bits_t, f8_bits_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <> struct cast_gt<b1x8_t, b1x8_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) const { return false; }
};

template <typename from_scalar_at> struct cast_gt<from_scalar_at, b1x8_t> {
    inline bool operator()(byte_t const* input, std::size_t dimensions, byte_t* output) const {
        from_scalar_at const* typed_input = reinterpret_cast<from_scalar_at const*>(input);
        unsigned char* typed_output = reinterpret_cast<unsigned char*>(output);
        for (std::size_t i = 0; i != dimensions; ++i)
            typed_output[i / CHAR_BIT] |= bool(typed_input[i]) ? (128 >> (i & (CHAR_BIT - 1))) : 0;
        return true;
    }
};

template <typename to_scalar_at> struct cast_gt<b1x8_t, to_scalar_at> {
    inline bool operator()(byte_t const* input, std::size_t dimensions, byte_t* output) const {
        unsigned char const* typed_input = reinterpret_cast<unsigned char const*>(input);
        to_scalar_at* typed_output = reinterpret_cast<to_scalar_at*>(output);
        for (std::size_t i = 0; i != dimensions; ++i)
            typed_output[i] = bool(typed_input[i / CHAR_BIT] & (128 >> (i & (CHAR_BIT - 1))));
        return true;
    }
};

} // namespace usearch
} // namespace unum
