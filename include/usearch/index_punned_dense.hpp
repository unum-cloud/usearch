#pragma once
#include <stdlib.h> // `aligned_alloc`

#include <functional>   // `std::function`
#include <numeric>      // `std::iota`
#include <shared_mutex> // `std::shared_mutex`
#include <thread>       // `std::thread`
#include <vector>       // `std::vector`

#include <usearch/index.hpp>
#include <usearch/index_punned_helpers.hpp>

#include <tsl/robin_map.h>

namespace unum {
namespace usearch {

struct cos_f8_t {
    using scalar_t = f8_bits_t;
    std::size_t dimensions;

    inline cos_f8_t(std::size_t dims) noexcept : dimensions(dims) {}
    inline punned_distance_t operator()(f8_bits_t const* a, f8_bits_t const* b) const noexcept {
        std::int32_t ab{}, a2{}, b2{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab, a2, b2)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dimensions; i++) {
            std::int16_t ai{a[i]};
            std::int16_t bi{b[i]};
            ab += ai * bi;
            a2 += square(ai);
            b2 += square(bi);
        }
        return (ab != 0) ? (1.f - ab / (std::sqrt(a2) * std::sqrt(b2))) : 0;
    }
};

struct l2sq_f8_t {
    using scalar_t = f8_bits_t;
    std::size_t dimensions;

    inline l2sq_f8_t(std::size_t dims) noexcept : dimensions(dims) {}
    inline punned_distance_t operator()(f8_bits_t const* a, f8_bits_t const* b) const noexcept {
        std::int32_t ab_deltas_sq{};
#if USEARCH_USE_OPENMP
#pragma omp simd reduction(+ : ab_deltas_sq)
#elif defined(USEARCH_DEFINED_CLANG)
#pragma clang loop vectorize(enable)
#elif defined(USEARCH_DEFINED_GCC)
#pragma GCC ivdep
#endif
        for (std::size_t i = 0; i != dimensions; i++)
            ab_deltas_sq += square(std::int16_t(a[i]) - std::int16_t(b[i]));
        return ab_deltas_sq;
    }
};

/**
 *  @brief  Oversimplified type-punned index for equidimensional vectors
 *          with automatic down-casting and hardware isa.
 */
template <typename label_at = std::int64_t, typename id_at = std::uint32_t> //
class index_punned_dense_gt {
  public:
    using label_t = label_at;
    using id_t = id_at;
    using distance_t = punned_distance_t;
    /// @brief Punned metric object.
    using metric_t = punned_metric_t;

  private:
    /// @brief Schema: input buffer, bytes in input buffer, output buffer.
    using cast_t = std::function<bool(byte_t const*, std::size_t, byte_t*)>;
    /// @brief Punned index.
    using index_t = index_gt<metric_t, label_t, id_t, byte_t, aligned_allocator_t, memory_mapping_allocator_t>;
    using index_allocator_t = aligned_allocator_gt<index_t, 64>;

    /// @brief A type-punned metric and metadata about present isa support.
    struct metric_and_meta_t {
        metric_t metric;
        isa_t isa;

        metric_and_meta_t(metric_t m = {}, isa_t i = isa_t::auto_k) : metric(m), isa(i) {}
        metric_and_meta_t(metric_and_meta_t const&) = default;
        metric_and_meta_t& operator=(metric_and_meta_t const&) = default;
    };

    using member_iterator_t = typename index_t::member_iterator_t;
    using member_citerator_t = typename index_t::member_citerator_t;
    using member_ref_t = typename index_t::member_ref_t;
    using member_cref_t = typename index_t::member_cref_t;

    /// @brief Number of unique dimensions in the vectors.
    std::size_t dimensions_ = 0;
    /// @brief Similar to `dimensions_`, but different for fraction byte-length scalars.
    std::size_t scalar_words_ = 0;
    std::size_t expansion_add_ = 0;
    std::size_t expansion_search_ = 0;
    index_t* typed_ = nullptr;

    std::size_t casted_vector_bytes_ = 0;
    scalar_kind_t scalar_kind_ = scalar_kind_t::f32_k;
    isa_t isa_ = isa_t::auto_k;

    mutable std::vector<byte_t> cast_buffer_;
    struct casts_t {
        cast_t from_b1x8;
        cast_t from_f8;
        cast_t from_f16;
        cast_t from_f32;
        cast_t from_f64;

        cast_t to_b1x8;
        cast_t to_f8;
        cast_t to_f16;
        cast_t to_f32;
        cast_t to_f64;
    } casts_;

    metric_t root_metric_;

    mutable std::vector<std::size_t> available_threads_;
    mutable std::mutex available_threads_mutex_;

    using shared_mutex_t = std::mutex; // TODO: Find an OS-compatible solution
    using shared_lock_t = std::unique_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    mutable shared_mutex_t lookup_table_mutex_;
    tsl::robin_map<label_t, id_t> lookup_table_;

  public:
    using search_result_t = typename index_t::search_result_t;
    using add_result_t = typename index_t::add_result_t;
    using serialization_result_t = typename index_t::serialization_result_t;
    using stats_t = typename index_t::stats_t;

    index_punned_dense_gt() = default;
    index_punned_dense_gt(index_punned_dense_gt&& other) { swap(other); }
    index_punned_dense_gt& operator=(index_punned_dense_gt&& other) {
        swap(other);
        return *this;
    }

    ~index_punned_dense_gt() { index_allocator_t{}.deallocate(typed_, 1); }

    void swap(index_punned_dense_gt& other) {
        std::swap(dimensions_, other.dimensions_);
        std::swap(scalar_words_, other.scalar_words_);
        std::swap(casted_vector_bytes_, other.casted_vector_bytes_);
        std::swap(scalar_kind_, other.scalar_kind_);
        std::swap(isa_, other.isa_);
        std::swap(typed_, other.typed_);
        std::swap(cast_buffer_, other.cast_buffer_);
        std::swap(casts_, other.casts_);
        std::swap(root_metric_, other.root_metric_);
        std::swap(available_threads_, other.available_threads_);
    }

    static index_config_t optimize(index_config_t config) { return index_t::optimize(config); }

    std::size_t dimensions() const { return dimensions_; }
    std::size_t scalar_words() const { return scalar_words_; }
    std::size_t connectivity() const { return typed_->connectivity(); }
    std::size_t size() const { return typed_->size(); }
    std::size_t capacity() const { return typed_->capacity(); }
    index_config_t const& config() const { return typed_->config(); }
    index_limits_t const& limits() const { return typed_->limits(); }
    void clear() { return typed_->clear(); }

    std::size_t expansion_add() const { return expansion_add_; }
    std::size_t expansion_search() const { return expansion_search_; }
    void change_expansion_add(std::size_t n) { expansion_add_ = n; }
    void change_expansion_search(std::size_t n) { expansion_search_ = n; }

    member_citerator_t cbegin() const { return typed_->cbegin(); }
    member_citerator_t cend() const { return typed_->cend(); }
    member_citerator_t begin() const { return typed_->begin(); }
    member_citerator_t end() const { return typed_->end(); }
    member_iterator_t begin() { return typed_->begin(); }
    member_iterator_t end() { return typed_->end(); }

    stats_t stats() const { return typed_->stats(); }
    stats_t stats(std::size_t level) const { return typed_->stats(level); }

    scalar_kind_t scalar_kind() const { return scalar_kind_; }
    isa_t isa() const { return isa_; }

    serialization_result_t save(char const* path) const { return typed_->save(path); }
    serialization_result_t load(char const* path) { return typed_->load(path); }
    serialization_result_t view(char const* path) { return typed_->view(path); }
    bool reserve(index_limits_t limits) { return typed_->reserve(limits); }

    std::size_t memory_usage(std::size_t allocator_entry_bytes = default_allocator_entry_bytes()) const {
        return typed_->memory_usage(allocator_entry_bytes);
    }

    bool contains(label_t label) const {
        shared_lock_t lock(lookup_table_mutex_);
        return lookup_table_.contains(label);
    }

    void export_labels(label_t* labels, std::size_t limit) const {
        shared_lock_t lock(lookup_table_mutex_);
        for (auto it = lookup_table_.begin(); it != lookup_table_.end() && limit; ++it, ++labels, --limit)
            *labels = it->first;
    }

    // clang-format off
    add_result_t add(label_t label, b1x8_t const* vector) { return add_(label, vector, casts_.from_b1x8); }
    add_result_t add(label_t label, f8_bits_t const* vector) { return add_(label, vector, casts_.from_f8); }
    add_result_t add(label_t label, f16_t const* vector) { return add_(label, vector, casts_.from_f16); }
    add_result_t add(label_t label, f32_t const* vector) { return add_(label, vector, casts_.from_f32); }
    add_result_t add(label_t label, f64_t const* vector) { return add_(label, vector, casts_.from_f64); }

    add_result_t add(label_t label, b1x8_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_b1x8); }
    add_result_t add(label_t label, f8_bits_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f8); }
    add_result_t add(label_t label, f16_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f16); }
    add_result_t add(label_t label, f32_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f32); }
    add_result_t add(label_t label, f64_t const* vector, add_config_t config) { return add_(label, vector, config, casts_.from_f64); }

    search_result_t search(b1x8_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_b1x8); }
    search_result_t search(f8_bits_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f8); }
    search_result_t search(f16_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f16); }
    search_result_t search(f32_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f32); }
    search_result_t search(f64_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f64); }

    search_result_t search(b1x8_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_b1x8); }
    search_result_t search(f8_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f8); }
    search_result_t search(f16_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f16); }
    search_result_t search(f32_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f32); }
    search_result_t search(f64_t const* vector, std::size_t wanted, search_config_t config) const { return search_(vector, wanted, config, casts_.from_f64); }

    search_result_t search_around(label_t hint, b1x8_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_b1x8); }
    search_result_t search_around(label_t hint, f8_bits_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f8); }
    search_result_t search_around(label_t hint, f16_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f16); }
    search_result_t search_around(label_t hint, f32_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f32); }
    search_result_t search_around(label_t hint, f64_t const* vector, std::size_t wanted) const { return search_around_(hint, vector, wanted, casts_.from_f64); }

    search_result_t search_around(label_t hint, b1x8_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_b1x8); }
    search_result_t search_around(label_t hint, f8_bits_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f8); }
    search_result_t search_around(label_t hint, f16_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f16); }
    search_result_t search_around(label_t hint, f32_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f32); }
    search_result_t search_around(label_t hint, f64_t const* vector, std::size_t wanted, search_config_t config) const { return search_around_(hint, vector, wanted, config, casts_.from_f64); }

    bool get(label_t label, b1x8_t* vector) const { return get_(label, vector, casts_.to_b1x8); }
    bool get(label_t label, f8_bits_t* vector) const { return get_(label, vector, casts_.to_f8); }
    bool get(label_t label, f16_t* vector) const { return get_(label, vector, casts_.to_f16); }
    bool get(label_t label, f32_t* vector) const { return get_(label, vector, casts_.to_f32); }
    bool get(label_t label, f64_t* vector) const { return get_(label, vector, casts_.to_f64); }
    // clang-format on

    static index_punned_dense_gt make(                       //
        std::size_t dimensions, metric_kind_t metric,        //
        index_config_t config = {},                          //
        scalar_kind_t accuracy = scalar_kind_t::f32_k,       //
        std::size_t expansion_add = default_expansion_add(), //
        std::size_t expansion_search = default_expansion_search()) {

        return make_(                                //
            dimensions, accuracy,                    //
            config, expansion_add, expansion_search, //
            make_metric_(metric, dimensions, accuracy), make_casts_(accuracy));
    }

    static index_punned_dense_gt make(                       //
        std::size_t dimensions, metric_t metric,             //
        index_config_t config = {},                          //
        scalar_kind_t accuracy = scalar_kind_t::f32_k,       //
        std::size_t expansion_add = default_expansion_add(), //
        std::size_t expansion_search = default_expansion_search()) {

        return make_(                                //
            dimensions, accuracy,                    //
            config, expansion_add, expansion_search, //
            metric_and_meta_t(metric), make_casts_(accuracy));
    }

    index_punned_dense_gt fork() const {
        index_punned_dense_gt result;

        result.dimensions_ = dimensions_;
        result.scalar_words_ = scalar_words_;
        result.scalar_kind_ = scalar_kind_;
        result.isa_ = isa_;
        result.casted_vector_bytes_ = casted_vector_bytes_;
        result.cast_buffer_ = cast_buffer_;
        result.casts_ = casts_;

        result.root_metric_ = root_metric_;
        index_t* raw = index_allocator_t{}.allocate(1);
        new (raw) index_t(config(), root_metric_);
        result.typed_ = raw;

        return result;
    }

  private:
    struct thread_lock_t {
        index_punned_dense_gt const& parent;
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
    search_result_t search_(                         //
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
    search_result_t search_around_(                                //
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

    template <typename scalar_at> bool get_(label_t label, scalar_at* reconstructed, cast_t const& cast) const {
        id_t id;
        {
            shared_lock_t lock(lookup_table_mutex_);
            auto it = lookup_table_.find(label);
            if (it == lookup_table_.end())
                return false;
            id = it->second;
        }
        member_citerator_t iterator = typed_->cbegin() + id;
        member_cref_t member = *iterator;
        byte_t const* casted_vector = reinterpret_cast<byte_t const*>(member.vector.data());
        bool casted = cast(casted_vector, casted_vector_bytes_, (byte_t*)reconstructed);
        if (!casted)
            std::memcpy(reconstructed, casted_vector, casted_vector_bytes_);
        return true;
    }

    template <typename scalar_at> add_result_t add_(label_t label, scalar_at const* vector, cast_t const& cast) {
        thread_lock_t lock = thread_lock_();
        add_config_t add_config;
        add_config.thread = lock.thread_id;
        return add_(label, vector, add_config, cast);
    }

    template <typename scalar_at>
    search_result_t search_(                         //
        scalar_at const* vector, std::size_t wanted, //
        cast_t const& cast) const {
        thread_lock_t lock = thread_lock_();
        search_config_t search_config;
        search_config.thread = lock.thread_id;
        return search_(vector, wanted, search_config, cast);
    }

    template <typename scalar_at>
    search_result_t search_around_(                                //
        label_t hint, scalar_at const* vector, std::size_t wanted, //
        cast_t const& cast) const {
        thread_lock_t lock = thread_lock_();
        search_config_t search_config;
        search_config.thread = lock.thread_id;
        return search_around_(hint, vector, wanted, search_config, cast);
    }

    static index_punned_dense_gt make_(                                                 //
        std::size_t dimensions, scalar_kind_t scalar_kind,                              //
        index_config_t config, std::size_t expansion_add, std::size_t expansion_search, //
        metric_and_meta_t metric_and_meta, casts_t casts) {

        std::size_t hardware_threads = std::thread::hardware_concurrency();
        index_punned_dense_gt result;
        result.dimensions_ = dimensions;
        result.scalar_kind_ = scalar_kind;
        result.scalar_words_ = count_scalar_words_(dimensions, scalar_kind);
        result.expansion_add_ = expansion_add;
        result.expansion_search_ = expansion_search;
        result.casted_vector_bytes_ = bytes_per_scalar(scalar_kind) * result.scalar_words_;
        result.cast_buffer_.resize(hardware_threads * result.casted_vector_bytes_);
        result.casts_ = casts;
        result.isa_ = metric_and_meta.isa;
        result.root_metric_ = metric_and_meta.metric;

        // Fill the thread IDs.
        result.available_threads_.resize(hardware_threads);
        std::iota(result.available_threads_.begin(), result.available_threads_.end(), 0ul);

        // Available since C11, but only C++17, so we use the C version.
        index_t* raw = index_allocator_t{}.allocate(1);
        new (raw) index_t(config, metric_and_meta.metric);
        result.typed_ = raw;
        return result;
    }

    template <typename to_scalar_at> static casts_t make_casts_() {
        casts_t result;

        result.from_b1x8 = cast_gt<b1x8_t, to_scalar_at>{};
        result.from_f8 = cast_gt<f8_bits_t, to_scalar_at>{};
        result.from_f16 = cast_gt<f16_t, to_scalar_at>{};
        result.from_f32 = cast_gt<f32_t, to_scalar_at>{};
        result.from_f64 = cast_gt<f64_t, to_scalar_at>{};

        result.to_b1x8 = cast_gt<to_scalar_at, b1x8_t>{};
        result.to_f8 = cast_gt<to_scalar_at, f8_bits_t>{};
        result.to_f16 = cast_gt<to_scalar_at, f16_t>{};
        result.to_f32 = cast_gt<to_scalar_at, f32_t>{};
        result.to_f64 = cast_gt<to_scalar_at, f64_t>{};

        return result;
    }

    static casts_t make_casts_(scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f64_k: return make_casts_<f64_t>();
        case scalar_kind_t::f32_k: return make_casts_<f32_t>();
        case scalar_kind_t::f16_k: return make_casts_<f16_t>();
        case scalar_kind_t::f8_k: return make_casts_<f8_bits_t>();
        case scalar_kind_t::b1x8_k: return make_casts_<b1x8_t>();
        default: return {};
        }
    }

    static std::size_t count_scalar_words_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f64_k: return dimensions;
        case scalar_kind_t::f32_k: return dimensions;
        case scalar_kind_t::f16_k: return dimensions;
        case scalar_kind_t::f8_k: return dimensions;
        case scalar_kind_t::b1x8_k: return divide_round_up<CHAR_BIT>(dimensions);
        default: return {};
        }
    }

    template <typename scalar_at, typename typed_at> static metric_t pun_metric_(typed_at metric) {
        using scalar_t = scalar_at;
        using vector_view_t = span_gt<scalar_t const>;
        return [=](punned_vector_view_t a, punned_vector_view_t b) -> punned_distance_t {
            std::size_t dims = a.size() / sizeof(scalar_t);
            vector_view_t a_typed{(scalar_t const*)a.data(), dims};
            vector_view_t b_typed{(scalar_t const*)b.data(), dims};
            return metric(a_typed, b_typed);
        };
    }

    static metric_and_meta_t ip_metric_f32_(std::size_t dimensions) {
        (void)dimensions;
#if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_t::avx2_k) && dimensions % 4 == 0)
            return {
                pun_metric_<simsimd_f32_t>([=](simsimd_f32_t const* a, simsimd_f32_t const* b) {
                    return 1.f - simsimd_dot_f32x4avx2(a, b, dimensions);
                }),
                isa_t::avx2_k,
            };
        if (hardware_supports(isa_t::sve_k))
            return {
                pun_metric_<simsimd_f32_t>([=](simsimd_f32_t const* a, simsimd_f32_t const* b) {
                    return 1.f - simsimd_dot_f32sve(a, b, dimensions);
                }),
                isa_t::sve_k,
            };
        if (hardware_supports(isa_t::neon_k) && dimensions % 4 == 0)
            return {
                pun_metric_<simsimd_f32_t>([=](simsimd_f32_t const* a, simsimd_f32_t const* b) {
                    return 1.f - simsimd_dot_f32x4neon(a, b, dimensions);
                }),
                isa_t::neon_k,
            };
#endif
        return pun_metric_<f32_t>(ip_gt<f32_t>{});
    }

    static metric_and_meta_t cos_metric_f16_(std::size_t dimensions) {
        (void)dimensions;
#if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_t::avx512_k) && dimensions % 16 == 0)
            return {
                pun_metric_<simsimd_f16_t>([=](simsimd_f16_t const* a, simsimd_f16_t const* b) {
                    return 1.f - simsimd_cos_f16x16avx512(a, b, dimensions);
                }),
                isa_t::avx512_k,
            };
        if (hardware_supports(isa_t::neon_k) && dimensions % 4 == 0)
            return {
                pun_metric_<simsimd_f16_t>([=](simsimd_f16_t const* a, simsimd_f16_t const* b) {
                    return 1.f - simsimd_cos_f16x4neon(a, b, dimensions);
                }),
                isa_t::neon_k,
            };
#endif
        return pun_metric_<f16_t>(cos_gt<f16_t, f32_t>{});
    }

    static metric_and_meta_t cos_metric_f8_(std::size_t dimensions) {
        (void)dimensions;
#if USEARCH_USE_SIMSIMD
        if (hardware_supports(isa_t::neon_k) && dimensions % 16 == 0)
            return {
                pun_metric_<int8_t>(
                    [=](int8_t const* a, int8_t const* b) { return 1.f - simsimd_cos_i8x16neon(a, b, dimensions); }),
                isa_t::neon_k,
            };
#endif
        return pun_metric_<f8_bits_t>(cos_f8_t{dimensions});
    }

    static metric_and_meta_t ip_metric_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f32_k:
            // The two most common numeric types for the most common metric have optimized versions
            return ip_metric_f32_(dimensions);
        case scalar_kind_t::f16_k:
            // Dot-product accumulates error, Cosine-distance normalizes it
            return cos_metric_f16_(dimensions);

        case scalar_kind_t::f8_k: return cos_metric_f8_(dimensions);
        case scalar_kind_t::f64_k: return pun_metric_<f64_t>(ip_gt<f64_t>{});
        default: return {};
        }
    }

    static metric_and_meta_t l2sq_metric_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return pun_metric_<f8_bits_t>(l2sq_f8_t{dimensions});
        case scalar_kind_t::f16_k: return pun_metric_<f16_t>(l2sq_gt<f16_t, f32_t>{});
        case scalar_kind_t::f32_k: return pun_metric_<f32_t>(l2sq_gt<f32_t>{});
        case scalar_kind_t::f64_k: return pun_metric_<f64_t>(l2sq_gt<f64_t>{});
        default: return {};
        }
    }

    static metric_and_meta_t cos_metric_(std::size_t dimensions, scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return cos_metric_f8_(dimensions);
        case scalar_kind_t::f16_k: return cos_metric_f16_(dimensions);
        case scalar_kind_t::f32_k: return pun_metric_<f32_t>(cos_gt<f32_t>{});
        case scalar_kind_t::f64_k: return pun_metric_<f64_t>(cos_gt<f64_t>{});
        default: return {};
        }
    }

    static metric_and_meta_t haversine_metric_(scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return pun_metric_<f8_bits_t>(haversine_gt<f8_bits_t, f32_t>{});
        case scalar_kind_t::f16_k: return pun_metric_<f16_t>(haversine_gt<f16_t, f32_t>{});
        case scalar_kind_t::f32_k: return pun_metric_<f32_t>(haversine_gt<f32_t>{});
        case scalar_kind_t::f64_k: return pun_metric_<f64_t>(haversine_gt<f64_t>{});
        default: return {};
        }
    }

    static metric_and_meta_t pearson_metric_(scalar_kind_t accuracy) {
        switch (accuracy) {
        case scalar_kind_t::f8_k: return pun_metric_<f8_bits_t>(pearson_correlation_gt<f8_bits_t, f32_t>{});
        case scalar_kind_t::f16_k: return pun_metric_<f16_t>(pearson_correlation_gt<f16_t, f32_t>{});
        case scalar_kind_t::f32_k: return pun_metric_<f32_t>(pearson_correlation_gt<f32_t>{});
        case scalar_kind_t::f64_k: return pun_metric_<f64_t>(pearson_correlation_gt<f64_t>{});
        default: return {};
        }
    }

    static metric_and_meta_t make_metric_(metric_kind_t kind, std::size_t dimensions, scalar_kind_t accuracy) {
        switch (kind) {
        case metric_kind_t::ip_k: return ip_metric_(dimensions, accuracy);
        case metric_kind_t::cos_k: return cos_metric_(dimensions, accuracy);
        case metric_kind_t::l2sq_k: return l2sq_metric_(dimensions, accuracy);
        case metric_kind_t::pearson_k: return pearson_metric_(accuracy);
        case metric_kind_t::haversine_k: return haversine_metric_(accuracy);
        case metric_kind_t::hamming_k: return pun_metric_<unsigned char>(hamming_gt<unsigned char>{});
        case metric_kind_t::jaccard_k: // Equivalent to Tanimoto
        case metric_kind_t::tanimoto_k: return pun_metric_<unsigned char>(tanimoto_gt<unsigned char>{});
        case metric_kind_t::sorensen_k: return pun_metric_<unsigned char>(sorensen_gt<unsigned char>{});
        default: return {};
        }
    }
};

using punned_small_t = index_punned_dense_gt<std::int64_t, std::uint32_t>;
using punned_big_t = index_punned_dense_gt<uuid_t, uint40_t>;

} // namespace usearch
} // namespace unum
