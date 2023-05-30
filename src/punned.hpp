#pragma once
#include <stdlib.h> // `aligned_alloc`

#include <functional>   // `std::function`
#include <numeric>      // `std::iota`
#include <shared_mutex> // `std::shared_mutex`
#include <stdexcept>    // `std::invalid_argument`
#include <thread>       // `std::thread`
#include <vector>       // `std::vector`

#include <fp16/fp16.h>
#include <tsl/robin_map.h>

#include <usearch/usearch.hpp>

#if defined(USEARCH_USE_OPENMP)
#include <omp.h> // `omp_get_num_threads()`
#endif

#if defined(USEARCH_IS_LINUX)
#include <sys/auxv.h>
#endif

#if defined(USEARCH_IS_ARM)
#include <arm_fp16.h>
#endif

#if defined(USEARCH_USE_SIMSIMD)
#include <simsimd/simsimd.h>
#endif

namespace unum {
namespace usearch {

using byte_t = char;
using punned_distance_t = float;
using punned_metric_t = punned_distance_t (*)(byte_t const*, byte_t const*, std::size_t, std::size_t);

template <typename metric_at>
punned_distance_t punned_metric(byte_t const* a, byte_t const* b, std::size_t a_bytes, std::size_t b_bytes) {
    using scalar_t = typename metric_at::scalar_t;
    return metric_at{}((scalar_t const*)a, (scalar_t const*)b, a_bytes / sizeof(scalar_t), b_bytes / sizeof(scalar_t));
}

using punned_stateful_metric_t =
    std::function<punned_distance_t(byte_t const*, byte_t const*, std::size_t, std::size_t)>;

template <typename at, typename compare_at> inline at clamp(at v, at lo, at hi, compare_at comp) {
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
template <typename at> inline at clamp(at v, at lo, at hi) { return usearch::clamp(v, lo, hi, std::less<at>{}); }

class f8_bits_t;
class f16_bits_t;

inline float f16_to_f32(std::uint16_t u16) {
#if defined(__AVX512F__)
    _Float16 f16;
    std::memcpy(&f16, &u16, sizeof(std::uint16_t));
    return float(f16);
#elif defined(USEARCH_IS_ARM)
    __fp16 f16;
    std::memcpy(&f16, &u16, sizeof(std::uint16_t));
    return float(f16);
#else
    return fp16_ieee_to_fp32_value(u16);
#endif
}

inline std::uint16_t f32_to_f16(float f32) {
#if defined(__AVX512F__)
    _Float16 f16 = _Float16(f32);
    std::uint16_t u16;
    std::memcpy(&u16, &f16, sizeof(std::uint16_t));
    return u16;
#elif defined(USEARCH_IS_ARM)
    __fp16 f16 = __fp16(f32);
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
    inline f16_bits_t() : uint16_(0) {}
    inline f16_bits_t(f16_bits_t&&) = default;
    inline f16_bits_t& operator=(f16_bits_t&&) = default;
    inline f16_bits_t(f16_bits_t const&) = default;
    inline f16_bits_t& operator=(f16_bits_t const&) = default;

    inline operator float() const { return f16_to_f32(uint16_); }

    inline f16_bits_t(f8_bits_t);
    inline f16_bits_t(float v) : uint16_(f32_to_f16(v)) {}
    inline f16_bits_t(double v) : uint16_(f32_to_f16(v)) {}

    inline f16_bits_t operator+(f16_bits_t other) const { return {float(*this) + float(other)}; }
    inline f16_bits_t operator-(f16_bits_t other) const { return {float(*this) - float(other)}; }
    inline f16_bits_t operator*(f16_bits_t other) const { return {float(*this) * float(other)}; }
    inline f16_bits_t operator/(f16_bits_t other) const { return {float(*this) / float(other)}; }
    inline f16_bits_t operator+(float other) const { return {float(*this) + other}; }
    inline f16_bits_t operator-(float other) const { return {float(*this) - other}; }
    inline f16_bits_t operator*(float other) const { return {float(*this) * other}; }
    inline f16_bits_t operator/(float other) const { return {float(*this) / other}; }
    inline f16_bits_t operator+(double other) const { return {float(*this) + other}; }
    inline f16_bits_t operator-(double other) const { return {float(*this) - other}; }
    inline f16_bits_t operator*(double other) const { return {float(*this) * other}; }
    inline f16_bits_t operator/(double other) const { return {float(*this) / other}; }

    inline f16_bits_t& operator+=(float v) {
        uint16_ = f32_to_f16(v + f16_to_f32(uint16_));
        return *this;
    }

    inline f16_bits_t& operator-=(float v) {
        uint16_ = f32_to_f16(v - f16_to_f32(uint16_));
        return *this;
    }

    inline f16_bits_t& operator*=(float v) {
        uint16_ = f32_to_f16(v * f16_to_f32(uint16_));
        return *this;
    }

    inline f16_bits_t& operator/=(float v) {
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

    inline f8_bits_t() : int8_(0) {}
    inline f8_bits_t(f8_bits_t&&) = default;
    inline f8_bits_t& operator=(f8_bits_t&&) = default;
    inline f8_bits_t(f8_bits_t const&) = default;
    inline f8_bits_t& operator=(f8_bits_t const&) = default;

    inline operator float() const { return float(int8_) / divisor_k; }
    inline operator f16_bits_t() const { return float(int8_) / divisor_k; }
    inline operator double() const { return double(int8_) / divisor_k; }
    inline explicit operator std::int8_t() const { return int8_; }
    inline explicit operator std::int16_t() const { return int8_; }
    inline explicit operator std::int32_t() const { return int8_; }
    inline explicit operator std::int64_t() const { return int8_; }

    inline f8_bits_t(float v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
    inline f8_bits_t(double v)
        : int8_(usearch::clamp<std::int8_t>(static_cast<std::int8_t>(v * divisor_k), min_k, max_k)) {}
};

inline f16_bits_t::f16_bits_t(f8_bits_t v) : f16_bits_t(float(v)) {}

struct cos_f8_t {
    punned_distance_t operator()(f8_bits_t const* a, f8_bits_t const* b, std::size_t n) const {
        std::int32_t ab{}, a2{}, b2{};
        for (std::size_t i = 0; i != n; i++) {
            std::int16_t ai{a[i]};
            std::int16_t bi{b[i]};
            ab += ai * bi;
            a2 += square(ai);
            b2 += square(bi);
        }
        return 1.f - ab / (std::sqrt(a2) * std::sqrt(b2));
    }
};

struct l2sq_f8_t {
    punned_distance_t operator()(f8_bits_t const* a, f8_bits_t const* b, std::size_t n) const {
        std::int32_t ab_deltas_sq{};
        for (std::size_t i = 0; i != n; i++)
            ab_deltas_sq += square(std::int16_t(a[i]) - std::int16_t(b[i]));
        return ab_deltas_sq;
    }
};

struct uuid_t {
    std::uint8_t octets[16];
};

template <typename callback_at> //
void multithreaded(std::size_t threads, std::size_t tasks, callback_at&& callback) {

    if (threads == 0)
        threads = std::thread::hardware_concurrency();
    if (threads == 1 || tasks <= 128) {
        for (std::size_t task_idx = 0; task_idx < tasks; ++task_idx)
            callback(0, task_idx);
        return;
    }

#if defined(USEARCH_USE_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < tasks; ++i)
        callback(omp_get_thread_num(), i);
#else
    std::vector<std::thread> threads_pool;
    std::size_t tasks_per_thread = (tasks / threads) + ((tasks % threads) != 0);
    for (std::size_t thread_idx = 0; thread_idx != threads; ++thread_idx) {
        threads_pool.emplace_back([=]() {
            for (std::size_t task_idx = thread_idx * tasks_per_thread;
                 task_idx < (std::min)(tasks, thread_idx * tasks_per_thread + tasks_per_thread); ++task_idx)
                callback(thread_idx, task_idx);
        });
    }
    for (std::size_t thread_idx = 0; thread_idx != threads; ++thread_idx)
        threads_pool[thread_idx].join();
#endif
}

/**
 *  @brief Relies on `posix_memalign` for aligned memory allocations.
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

    pointer allocate(size_type length) const {
        void* result = nullptr;
        int status = posix_memalign(&result, alignment_ak, ceil2(length * sizeof(value_type)));
        return status == 0 ? (pointer)result : nullptr;
    }

    void deallocate(pointer begin, size_type) const { std::free(begin); }
};

using aligned_allocator_t = aligned_allocator_gt<>;

enum class accuracy_t {
    f32_k,
    f16_k,
    f64_k,
    f8_k,
};

inline char const* accuracy_name(accuracy_t accuracy) {
    switch (accuracy) {
    case accuracy_t::f32_k: return "f32";
    case accuracy_t::f16_k: return "f16";
    case accuracy_t::f64_k: return "f64";
    case accuracy_t::f8_k: return "f8";
    default: return "";
    }
}

inline bool str_equals(char const* begin, std::size_t len, char const* other_begin) {
    std::size_t other_len = strlen(other_begin);
    return len == other_len && strncmp(begin, other_begin, len) == 0;
}

inline accuracy_t accuracy_from_name(char const* name, std::size_t len) {
    accuracy_t accuracy;
    if (str_equals(name, len, "f32"))
        accuracy = accuracy_t::f32_k;
    else if (str_equals(name, len, "f64"))
        accuracy = accuracy_t::f64_k;
    else if (str_equals(name, len, "f16"))
        accuracy = accuracy_t::f16_k;
    else if (str_equals(name, len, "f8"))
        accuracy = accuracy_t::f8_k;
    else
        throw std::invalid_argument("Unknown type, choose: f32, f16, f64, f8");
    return accuracy;
}

template <typename index_at>
inline index_at index_from_name( //
    char const* name, std::size_t len, std::size_t dimensions, accuracy_t accuracy, config_t const& config) {

    if (str_equals(name, len, "l2sq") || str_equals(name, len, "euclidean_sq")) {
        if (dimensions == 0)
            throw std::invalid_argument("The number of dimensions must be positive");
        return index_at::l2sq(dimensions, accuracy, config);
    } else if (str_equals(name, len, "ip") || str_equals(name, len, "inner") || str_equals(name, len, "dot")) {
        if (dimensions == 0)
            throw std::invalid_argument("The number of dimensions must be positive");
        return index_at::ip(dimensions, accuracy, config);
    } else if (str_equals(name, len, "cos") || str_equals(name, len, "angular")) {
        if (dimensions == 0)
            throw std::invalid_argument("The number of dimensions must be positive");
        return index_at::cos(dimensions, accuracy, config);
    } else if (str_equals(name, len, "haversine")) {
        if (dimensions != 2 && dimensions != 0)
            throw std::invalid_argument("The number of dimensions must be equal to two");
        return index_at::haversine(accuracy, config);
    } else
        throw std::invalid_argument("Unknown distance, choose: l2sq, ip, cos, hamming, jaccard");
    return {};
}

enum class isa_t {
    auto_k,
    neon_k,
    sve_k,
    avx2_k,
    avx512_k,
};

inline char const* isa_name(isa_t isa) {
    switch (isa) {
    case isa_t::auto_k: return "auto";
    case isa_t::neon_k: return "neon";
    case isa_t::sve_k: return "sve";
    case isa_t::avx2_k: return "avx2";
    case isa_t::avx512_k: return "avx512";
    default: return "";
    }
}

inline std::size_t bytes_per_scalar(accuracy_t accuracy) {
    switch (accuracy) {
    case accuracy_t::f32_k: return 4;
    case accuracy_t::f16_k: return 2;
    case accuracy_t::f64_k: return 8;
    case accuracy_t::f8_k: return 1;
    default: return 0;
    }
}

inline bool supports_arm_sve() {
#if defined(USEARCH_IS_ARM)
#if defined(USEARCH_IS_LINUX)
    unsigned long capabilities = getauxval(AT_HWCAP);
    if (capabilities & HWCAP_SVE)
        return true;
#endif
#endif
    return false;
}

template <typename from_scalar_at, typename to_scalar_at> struct cast_gt {
    bool operator()(byte_t const* input, std::size_t bytes_in_input, byte_t* output) const {
        from_scalar_at const* typed_input = reinterpret_cast<from_scalar_at const*>(input);
        to_scalar_at* typed_output = reinterpret_cast<to_scalar_at*>(output);
        std::transform( //
            typed_input, typed_input + bytes_in_input / sizeof(from_scalar_at), typed_output,
            [](from_scalar_at from) { return to_scalar_at(from); });
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

/**
 *  @brief  Oversimplified type-punned index for equidimensional floating-point
 *          vectors with automatic down-casting and hardware acceleration.
 */
template <typename label_at = std::int64_t, typename id_at = std::uint32_t> //
class punned_gt {
  public:
    using label_t = label_at;
    using id_t = id_at;
    using distance_t = punned_distance_t;

  private:
    /// @brief Schema: input buffer, bytes in input buffer, output buffer.
    using cast_t = std::function<bool(byte_t const*, std::size_t, byte_t*)>;
    /// @brief Punned index.
    using index_t = index_gt<punned_stateful_metric_t, label_t, id_t, byte_t>;
    /// @brief A type-punned metric and metadata about present isa support.
    struct metric_and_meta_t {
        punned_stateful_metric_t metric;
        isa_t acceleration;
    };

    using member_iterator_t = typename index_t::member_iterator_t;
    using member_citerator_t = typename index_t::member_citerator_t;
    using member_ref_t = typename index_t::member_ref_t;
    using member_cref_t = typename index_t::member_cref_t;

    std::size_t dimensions_ = 0;
    std::size_t casted_vector_bytes_ = 0;
    accuracy_t accuracy_ = accuracy_t::f32_k;
    isa_t acceleration_ = isa_t::auto_k;

    index_t* typed_{};
    mutable std::vector<byte_t> cast_buffer_{};
    struct casts_t {
        cast_t from_f8{};
        cast_t from_f16{};
        cast_t from_f32{};
        cast_t from_f64{};
        cast_t to_f8{};
        cast_t to_f16{};
        cast_t to_f32{};
        cast_t to_f64{};
    } casts_;

    punned_stateful_metric_t root_metric_;

    mutable std::vector<std::size_t> available_threads_;
    mutable std::mutex available_threads_mutex_;

    using shared_mutex_t = std::mutex; // TODO: Find an OS-compatible solution
    using shared_lock_t = std::unique_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    mutable shared_mutex_t lookup_table_mutex_;
    tsl::robin_map<label_t, id_t> lookup_table_;

  public:
    using search_results_t = typename index_t::search_results_t;
    using add_result_t = typename index_t::add_result_t;
    using stats_t = typename index_t::stats_t;

    punned_gt() = default;
    punned_gt(punned_gt&& other) { swap(other); }
    punned_gt& operator=(punned_gt&& other) {
        swap(other);
        return *this;
    }

    ~punned_gt() { aligned_index_free_(typed_); }

    void swap(punned_gt& other) {
        std::swap(dimensions_, other.dimensions_);
        std::swap(casted_vector_bytes_, other.casted_vector_bytes_);
        std::swap(accuracy_, other.accuracy_);
        std::swap(acceleration_, other.acceleration_);
        std::swap(typed_, other.typed_);
        std::swap(cast_buffer_, other.cast_buffer_);
        std::swap(casts_, other.casts_);
        std::swap(root_metric_, other.root_metric_);
        std::swap(available_threads_, other.available_threads_);
    }

    static config_t optimize(config_t config) noexcept { return index_t::optimize(config); }

    std::size_t dimensions() const { return dimensions_; }
    std::size_t connectivity() const { return typed_->connectivity(); }
    std::size_t size() const { return typed_->size(); }
    std::size_t capacity() const { return typed_->capacity(); }
    config_t const& config() const { return typed_->config(); }
    void clear() { return typed_->clear(); }
    void change_expansion_add(std::size_t n) noexcept { typed_->change_expansion_add(n); }
    void change_expansion_search(std::size_t n) noexcept { typed_->change_expansion_search(n); }

    member_citerator_t cbegin() const noexcept { return typed_->cbegin(); }
    member_citerator_t cend() const noexcept { return typed_->cend(); }
    member_citerator_t begin() const noexcept { return typed_->begin(); }
    member_citerator_t end() const noexcept { return typed_->end(); }
    member_iterator_t begin() noexcept { return typed_->begin(); }
    member_iterator_t end() noexcept { return typed_->end(); }

    stats_t stats() const noexcept { return typed_->stats(); }
    stats_t stats(std::size_t level) const noexcept { return typed_->stats(level); }

    accuracy_t accuracy() const { return accuracy_; }
    isa_t acceleration() const { return acceleration_; }

    void save(char const* path) const { typed_->save(path); }
    void load(char const* path) { typed_->load(path); }
    void view(char const* path) { typed_->view(path); }
    void reserve(std::size_t capacity) { typed_->reserve(capacity); }

    std::size_t memory_usage(std::size_t allocator_entry_bytes = default_allocator_entry_bytes()) const noexcept {
        return typed_->memory_usage(allocator_entry_bytes);
    }

    // clang-format off
    add_result_t add(label_t label, f8_bits_t const* vector) { return add_(label, vector, casts_.from_f8); }
    add_result_t add(label_t label, f16_bits_t const* vector) { return add_(label, vector, casts_.from_f16); }
    add_result_t add(label_t label, f32_t const* vector) { return add_(label, vector, casts_.from_f32); }
    add_result_t add(label_t label, f64_t const* vector) { return add_(label, vector, casts_.from_f64); }

    add_result_t add(label_t label, f8_bits_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f8); }
    add_result_t add(label_t label, f16_bits_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f16); }
    add_result_t add(label_t label, f32_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f32); }
    add_result_t add(label_t label, f64_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f64); }

    search_results_t search(f8_bits_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f8); }
    search_results_t search(f16_bits_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f16); }
    search_results_t search(f32_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f32); }
    search_results_t search(f64_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f64); }

    search_results_t search(f8_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f8); }
    search_results_t search(f16_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f16); }
    search_results_t search(f32_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f32); }
    search_results_t search(f64_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f64); }

    search_results_t search_around(label_t hint, f8_bits_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f8); }
    search_results_t search_around(label_t hint, f16_bits_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f16); }
    search_results_t search_around(label_t hint, f32_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f32); }
    search_results_t search_around(label_t hint, f64_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f64); }

    search_results_t search_around(label_t hint, f8_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f8); }
    search_results_t search_around(label_t hint, f16_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f16); }
    search_results_t search_around(label_t hint, f32_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f32); }
    search_results_t search_around(label_t hint, f64_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f64); }

    void reconstruct(label_t label, f8_bits_t* vector) const { return reconstruct_(label, vector, casts_.to_f8); }
    void reconstruct(label_t label, f16_bits_t* vector) const { return reconstruct_(label, vector, casts_.to_f16); }
    void reconstruct(label_t label, f32_t* vector) const { return reconstruct_(label, vector, casts_.to_f32); }
    void reconstruct(label_t label, f64_t* vector) const { return reconstruct_(label, vector, casts_.to_f64); }

    static punned_gt ip(std::size_t dimensions, accuracy_t accuracy = accuracy_t::f16_k, config_t config = {}) { return make_(dimensions, accuracy, ip_metric_(dimensions, accuracy), make_casts_(accuracy), config); }
    static punned_gt l2sq(std::size_t dimensions, accuracy_t accuracy = accuracy_t::f16_k, config_t config = {}) { return make_(dimensions, accuracy, l2_metric_(dimensions, accuracy), make_casts_(accuracy), config); }
    static punned_gt cos(std::size_t dimensions, accuracy_t accuracy = accuracy_t::f32_k, config_t config = {}) { return make_(dimensions, accuracy, cos_metric_(dimensions, accuracy), make_casts_(accuracy), config); }
    static punned_gt haversine(accuracy_t accuracy = accuracy_t::f32_k, config_t config = {}) { return make_(2, accuracy, haversine_metric_(accuracy), make_casts_(accuracy), config); }
    // clang-format on

    static punned_gt udf(                                        //
        std::size_t dimensions, punned_stateful_metric_t metric, //
        accuracy_t accuracy = accuracy_t::f32_k, config_t config = {}) {
        return make_(dimensions, accuracy, {metric, isa_t::auto_k}, make_casts_(accuracy), config);
    }

    punned_gt fork() const {
        punned_gt result;

        result.dimensions_ = dimensions_;
        result.accuracy_ = accuracy_;
        result.acceleration_ = acceleration_;
        result.casted_vector_bytes_ = casted_vector_bytes_;
        result.cast_buffer_ = cast_buffer_;
        result.casts_ = casts_;

        result.root_metric_ = root_metric_;
        index_t* raw = aligned_index_alloc_();
        new (raw) index_t(config(), root_metric_);
        result.typed_ = raw;

        return result;
    }

  private:
    static index_t* aligned_index_alloc_() {
#if defined(USEARCH_IS_WINDOWS)
        return (index_t*)_aligned_malloc(64 * divide_round_up<64>(sizeof(index_t)), 64);
#else
        return (index_t*)aligned_alloc(64, 64 * divide_round_up<64>(sizeof(index_t)));
#endif
    }

    static void aligned_index_free_(index_t* raw) {
#if defined(USEARCH_IS_WINDOWS)
        _aligned_free(raw);
#else
        free(raw);
#endif
    }

    struct thread_lock_t {
        punned_gt const& parent;
        std::size_t thread_id;

        ~thread_lock_t() { parent.thread_unlock_(thread_id); }
    };

    thread_lock_t thread_lock_() const {
        available_threads_mutex_.lock();
        std::size_t thread_id = available_threads_.back();
        available_threads_.pop_back();
        available_threads_mutex_.unlock();
        return {*this, thread_id};
    }

    void thread_unlock_(std::size_t thread_id) const {
        available_threads_mutex_.lock();
        available_threads_.push_back(thread_id);
        available_threads_mutex_.unlock();
    }

    template <typename scalar_at>
    add_result_t add_(label_t label, scalar_at const* vector, add_config_t config, cast_t const& cast) {
        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(scalar_at);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * config.thread;
        bool casted = cast(vector_data, vector_bytes, casted_data);
        if (casted)
            vector_data = casted_data, vector_bytes = casted_vector_bytes_, config.store_vector = true;

        add_result_t result = typed_->add(label, {vector_data, vector_bytes}, config);
        {
            unique_lock_t lock(lookup_table_mutex_);
            lookup_table_.emplace(label, result.id);
        }
        return result;
    }

    template <typename scalar_at>
    search_results_t search_(                        //
        scalar_at const* vector, std::size_t wanted, //
        search_config_t config, cast_t const& cast) const {

        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(scalar_at);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * config.thread;
        bool casted = cast(vector_data, vector_bytes, casted_data);
        if (casted)
            vector_data = casted_data, vector_bytes = casted_vector_bytes_;

        return typed_->search({vector_data, vector_bytes}, wanted, config);
    }

    template <typename scalar_at>
    search_results_t search_around_(                               //
        label_t hint, scalar_at const* vector, std::size_t wanted, //
        search_config_t config, cast_t const& cast) const {

        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(scalar_at);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * config.thread;
        bool casted = cast(vector_data, vector_bytes, casted_data);
        if (casted)
            vector_data = casted_data, vector_bytes = casted_vector_bytes_;

        return typed_->search_around(static_cast<id_t>(hint), {vector_data, vector_bytes}, wanted, config);
    }

    id_t lookup_id_(label_t label) const {
        shared_lock_t lock(lookup_table_mutex_);
        return lookup_table_.at(label);
    }

    template <typename scalar_at> void reconstruct_(label_t label, scalar_at* reconstructed, cast_t const& cast) const {
        id_t id = lookup_id_(label);
        member_citerator_t iterator = typed_->cbegin() + id;
        member_cref_t member = *iterator;
        byte_t const* casted_vector = reinterpret_cast<byte_t const*>(member.vector.data());
        bool casted = cast(casted_vector, casted_vector_bytes_, (byte_t*)reconstructed);
        if (!casted)
            std::memcpy(reconstructed, casted_vector, casted_vector_bytes_);
    }

    template <typename scalar_at> add_result_t add_(label_t label, scalar_at const* vector, cast_t const& cast) {
        thread_lock_t lock = thread_lock_();
        add_config_t add_config;
        add_config.thread = lock.thread_id;
        return add_(label, vector, add_config, cast);
    }

    template <typename scalar_at>
    search_results_t search_(                        //
        scalar_at const* vector, std::size_t wanted, //
        cast_t const& cast) const {
        thread_lock_t lock = thread_lock_();
        search_config_t search_config;
        search_config.thread = lock.thread_id;
        return search_(vector, wanted, search_config, cast);
    }

    template <typename scalar_at>
    search_results_t search_around_(                               //
        label_t hint, scalar_at const* vector, std::size_t wanted, //
        cast_t const& cast) const {
        thread_lock_t lock = thread_lock_();
        search_config_t search_config;
        search_config.thread = lock.thread_id;
        return search_around_(hint, vector, wanted, search_config, cast);
    }

    static punned_gt make_(                               //
        std::size_t dimensions, accuracy_t accuracy,      //
        metric_and_meta_t metric_and_meta, casts_t casts, //
        config_t config) {

        std::size_t hardware_threads = std::thread::hardware_concurrency();
        config.max_threads_add = config.max_threads_add ? config.max_threads_add : hardware_threads;
        config.max_threads_search = config.max_threads_search ? config.max_threads_search : hardware_threads;
        std::size_t max_threads = (std::max)(config.max_threads_add, config.max_threads_search);
        punned_gt result;
        result.dimensions_ = dimensions;
        result.accuracy_ = accuracy;
        result.casted_vector_bytes_ = bytes_per_scalar(accuracy) * dimensions;
        result.cast_buffer_.resize(max_threads * result.casted_vector_bytes_);
        result.casts_ = casts;
        result.acceleration_ = metric_and_meta.acceleration;
        result.root_metric_ = metric_and_meta.metric;

        // Fill the thread IDs.
        result.available_threads_.resize(max_threads);
        std::iota(result.available_threads_.begin(), result.available_threads_.end(), 0ul);

        // Available since C11, but only C++17, so we use the C version.
        index_t* raw = aligned_index_alloc_();
        new (raw) index_t(config, metric_and_meta.metric);
        result.typed_ = raw;
        return result;
    }

    template <typename to_scalar_at> static casts_t make_casts_() {
        casts_t result;
        result.from_f8 = cast_gt<f8_bits_t, to_scalar_at>{};
        result.from_f16 = cast_gt<f16_bits_t, to_scalar_at>{};
        result.from_f32 = cast_gt<f32_t, to_scalar_at>{};
        result.from_f64 = cast_gt<f64_t, to_scalar_at>{};
        result.to_f8 = cast_gt<to_scalar_at, f8_bits_t>{};
        result.to_f16 = cast_gt<to_scalar_at, f16_bits_t>{};
        result.to_f32 = cast_gt<to_scalar_at, f32_t>{};
        result.to_f64 = cast_gt<to_scalar_at, f64_t>{};
        return result;
    }

    static casts_t make_casts_(accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::f64_k: return make_casts_<f64_t>();
        case accuracy_t::f32_k: return make_casts_<f32_t>();
        case accuracy_t::f16_k: return make_casts_<f16_bits_t>();
        case accuracy_t::f8_k: return make_casts_<f8_bits_t>();
        default: return {};
        }
    }

    template <typename scalar_at, typename typed_at> //
    static punned_stateful_metric_t pun_metric_(typed_at metric) {
        return [=](byte_t const* a, byte_t const* b, std::size_t bytes, std::size_t) -> float {
            return metric((scalar_at const*)a, (scalar_at const*)b, bytes / sizeof(scalar_at));
        };
    }

    static metric_and_meta_t ip_metric_f32_(std::size_t dimensions) {
        (void)dimensions;
#if defined(USEARCH_USE_SIMSIMD)
#if defined(USEARCH_IS_X86)
        if (dimensions % 4 == 0)
            return {
                pun_metric_<simsimd_f32_t>([](simsimd_f32_t const* a, simsimd_f32_t const* b, size_t d) {
                    return 1.f - simsimd_dot_f32x4avx2(a, b, d);
                }),
                isa_t::avx2_k,
            };
#elif defined(USEARCH_IS_ARM)
        if (supports_arm_sve())
            return {
                pun_metric_<simsimd_f32_t>([](simsimd_f32_t const* a, simsimd_f32_t const* b, size_t d) {
                    return 1.f - simsimd_dot_f32sve(a, b, d);
                }),
                isa_t::sve_k,
            };
        if (dimensions % 4 == 0)
            return {
                pun_metric_<simsimd_f32_t>([](simsimd_f32_t const* a, simsimd_f32_t const* b, size_t d) {
                    return 1.f - simsimd_dot_f32x4neon(a, b, d);
                }),
                isa_t::neon_k,
            };
#endif
#endif
        return {pun_metric_<f32_t>(ip_gt<f32_t>{}), isa_t::auto_k};
    }

    static metric_and_meta_t cos_metric_f16_(std::size_t dimensions) {
        (void)dimensions;
#if defined(USEARCH_USE_SIMSIMD)
#if defined(USEARCH_IS_X86)
        if (dimensions % 16 == 0)
            return {
                pun_metric_<simsimd_f16_t>([](simsimd_f16_t const* a, simsimd_f16_t const* b, size_t d) {
                    return 1.f - simsimd_cos_f16x16avx512(a, b, d);
                }),
                isa_t::avx512_k,
            };
#elif defined(USEARCH_IS_ARM)
        if (dimensions % 4 == 0)
            return {
                pun_metric_<simsimd_f16_t>([](simsimd_f16_t const* a, simsimd_f16_t const* b, size_t d) {
                    return 1.f - simsimd_cos_f16x4neon(a, b, d);
                }),
                isa_t::neon_k,
            };
#endif
#endif
        return {pun_metric_<f16_bits_t>(cos_gt<f16_bits_t, f32_t>{}), isa_t::auto_k};
    }

    static metric_and_meta_t ip_metric_(std::size_t dimensions, accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::f32_k:
            // The two most common numeric types for the most common metric have optimized versions
            return ip_metric_f32_(dimensions);
        case accuracy_t::f16_k:
            // Dot-product accumulates error, Cosine-distance normalizes it
            return cos_metric_f16_(dimensions);

        case accuracy_t::f8_k: return {pun_metric_<f8_bits_t>(cos_f8_t{}), isa_t::auto_k};
        case accuracy_t::f64_k: return {pun_metric_<f64_t>(ip_gt<f64_t>{}), isa_t::auto_k};
        default: return {};
        }
    }

    static metric_and_meta_t l2_metric_(std::size_t, accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::f8_k: return {pun_metric_<f8_bits_t>(l2sq_f8_t{}), isa_t::auto_k};
        case accuracy_t::f16_k: return {pun_metric_<f16_bits_t>(l2sq_gt<f16_bits_t, f32_t>{}), isa_t::auto_k};
        case accuracy_t::f32_k: return {pun_metric_<f32_t>(l2sq_gt<f32_t>{}), isa_t::auto_k};
        case accuracy_t::f64_k: return {pun_metric_<f64_t>(l2sq_gt<f64_t>{}), isa_t::auto_k};
        default: return {};
        }
    }

    static metric_and_meta_t cos_metric_(std::size_t dimensions, accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::f8_k: return {pun_metric_<f8_bits_t>(cos_f8_t{}), isa_t::auto_k};
        case accuracy_t::f16_k: return cos_metric_f16_(dimensions);
        case accuracy_t::f32_k: return {pun_metric_<f32_t>(cos_gt<f32_t>{}), isa_t::auto_k};
        case accuracy_t::f64_k: return {pun_metric_<f64_t>(cos_gt<f64_t>{}), isa_t::auto_k};
        default: return {};
        }
    }

    static metric_and_meta_t haversine_metric_(accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::f8_k: return {pun_metric_<f8_bits_t>(haversine_gt<f8_bits_t, f32_t>{}), isa_t::auto_k};
        case accuracy_t::f16_k: return {pun_metric_<f16_bits_t>(haversine_gt<f16_bits_t, f32_t>{}), isa_t::auto_k};
        case accuracy_t::f32_k: return {pun_metric_<f32_t>(haversine_gt<f32_t>{}), isa_t::auto_k};
        case accuracy_t::f64_k: return {pun_metric_<f64_t>(haversine_gt<f64_t>{}), isa_t::auto_k};
        default: return {};
        }
    }
};

using punned_small_t = punned_gt<std::int64_t, std::uint32_t>;
using punned_big_t = punned_gt<uuid_t, uint40_t>;

} // namespace usearch
} // namespace unum
