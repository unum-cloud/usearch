#pragma once
#include <stdlib.h> // `aligned_alloc`

#include <functional>   // `std::function`
#include <numeric>      // `std::iota`
#include <shared_mutex> // `std::shared_mutex`
#include <thread>       // `std::thread`
#include <vector>       // `std::vector`

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include <tsl/robin_set.h>

namespace unum {
namespace usearch {

template <typename, typename> class index_dense_gt;

// clang-format off
template <typename key_at, typename std::enable_if<std::is_integral<key_at>::value>::type* = nullptr> key_at default_free_value() { return std::numeric_limits<key_at>::max(); }
template <typename key_at, typename std::enable_if<std::is_same<key_at, uint40_t>::value>::type* = nullptr> uint40_t default_free_value() { return uint40_t::max(); }
template <typename key_at, typename std::enable_if<!std::is_integral<key_at>::value && !std::is_same<key_at, uint40_t>::value>::type* = nullptr> key_at default_free_value() { return key_at(); }
// clang-format on

/**
 *  @brief  The "magic" sequence helps infer the type of the file.
 *          USearch indexes start with the "usearch" string.
 */
constexpr char const* default_magic() { return "usearch"; }

using index_dense_head_buffer_t = byte_t[64];

static_assert(sizeof(index_dense_head_buffer_t) == 64, "File header should be exactly 64 bytes");

/**
 *  @brief  Serialized binary representations of the USearch index start with metadata.
 *          Metadata is parsed into a `index_dense_head_t`, containing the USearch package version,
 *          and the properties of the index.
 *
 *  It uses: 13 bytes for file versioning, 22 bytes for structural information = 35 bytes.
 *  The following 24 bytes contain binary size of the graph, of the vectors, and the checksum,
 *  leaving 5 bytes at the end vacant.
 */
struct index_dense_head_t {

    // Versioning:
    using magic_t = char[7];
    using version_t = std::uint16_t;

    // Versioning: 7 + 2 * 3 = 13 bytes
    char const* magic;
    misaligned_ref_gt<version_t> version_major;
    misaligned_ref_gt<version_t> version_minor;
    misaligned_ref_gt<version_t> version_patch;

    // Structural: 4 * 3 = 12 bytes
    misaligned_ref_gt<metric_kind_t> kind_metric;
    misaligned_ref_gt<scalar_kind_t> kind_scalar;
    misaligned_ref_gt<scalar_kind_t> kind_key;
    misaligned_ref_gt<scalar_kind_t> kind_compressed_slot;

    // Population: 8 * 3 = 24 bytes
    misaligned_ref_gt<std::uint64_t> count_present;
    misaligned_ref_gt<std::uint64_t> count_deleted;
    misaligned_ref_gt<std::uint64_t> dimensions;

    index_dense_head_t(byte_t* ptr) noexcept
        : magic((char const*)exchange(ptr, ptr + sizeof(magic_t))), //
          version_major(exchange(ptr, ptr + sizeof(version_t))),    //
          version_minor(exchange(ptr, ptr + sizeof(version_t))),    //
          version_patch(exchange(ptr, ptr + sizeof(version_t))),    //
          kind_metric(exchange(ptr, ptr + sizeof(metric_kind_t))),  //
          kind_scalar(exchange(ptr, ptr + sizeof(scalar_kind_t))),  //
          kind_key(exchange(ptr, ptr + sizeof(scalar_kind_t))),     //
          kind_compressed_slot(exchange(ptr, ptr + sizeof(scalar_kind_t))),
          count_present(exchange(ptr, ptr + sizeof(std::uint64_t))),
          count_deleted(exchange(ptr, ptr + sizeof(std::uint64_t))),
          dimensions(exchange(ptr, ptr + sizeof(std::uint64_t))) {}
};

struct index_dense_head_result_t {

    index_dense_head_buffer_t buffer;
    index_dense_head_t head;
    error_t error;

    explicit operator bool() const noexcept { return !error; }
    index_dense_head_result_t failed(error_t message) noexcept {
        error = std::move(message);
        return std::move(*this);
    }
};

struct index_dense_config_t : public index_config_t {
    std::size_t expansion_add = default_expansion_add();
    std::size_t expansion_search = default_expansion_search();
    bool exclude_vectors = false;
    bool allow_key_collisions = false;

    index_dense_config_t(index_config_t base) noexcept : index_config_t(base) {}

    index_dense_config_t(                                  //
        std::size_t connectivity = default_connectivity(), //
        std::size_t expansion_add = default_expansion_add(),
        std::size_t expansion_search = default_expansion_search()) noexcept
        : index_config_t(connectivity), //
          expansion_add(expansion_add ? expansion_add : default_expansion_add()),
          expansion_search(expansion_search ? expansion_search : default_expansion_search()) {}
};

struct index_dense_serialization_config_t {
    bool exclude_vectors = false;
    bool use_64_bit_dimensions = false;
};

struct index_dense_update_config_t : public index_update_config_t {
    bool force_vector_copy = false;

    index_dense_update_config_t() = default;
    index_dense_update_config_t(index_update_config_t base) noexcept : index_update_config_t(base) {}
};

struct index_dense_copy_config_t : public index_copy_config_t {
    bool force_vector_copy = true;

    index_dense_copy_config_t() = default;
    index_dense_copy_config_t(index_copy_config_t base) noexcept : index_copy_config_t(base) {}
};

struct index_dense_metadata_result_t {
    index_dense_serialization_config_t config;
    index_dense_head_buffer_t head_buffer;
    index_dense_head_t head;
    error_t error;

    explicit operator bool() const noexcept { return !error; }
    index_dense_metadata_result_t failed(error_t message) noexcept {
        error = std::move(message);
        return std::move(*this);
    }

    index_dense_metadata_result_t() noexcept : config(), head_buffer(), head(head_buffer), error() {}

    index_dense_metadata_result_t(index_dense_metadata_result_t&& other) noexcept
        : config(), head_buffer(), head(head_buffer), error(std::move(other.error)) {
        std::memcpy(&config, &other.config, sizeof(other.config));
        std::memcpy(&head_buffer, &other.head_buffer, sizeof(other.head_buffer));
    }

    index_dense_metadata_result_t& operator=(index_dense_metadata_result_t&& other) noexcept {
        std::memcpy(&config, &other.config, sizeof(other.config));
        std::memcpy(&head_buffer, &other.head_buffer, sizeof(other.head_buffer));
        error = std::move(other.error);
        return *this;
    }
};

/**
 *  @brief  Extracts metadata from pre-constructed index on disk,
 *          without loading it or mapping the whole binary file.
 */
inline index_dense_metadata_result_t index_dense_metadata(char const* file_path) noexcept {
    index_dense_metadata_result_t result;
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> file(std::fopen(file_path, "rb"), &std::fclose);
    if (!file)
        return result.failed(std::strerror(errno));

    // Read the header
    std::size_t read = std::fread(result.head_buffer, sizeof(index_dense_head_buffer_t), 1, file.get());
    if (!read)
        return result.failed(std::feof(file.get()) ? "End of file reached!" : std::strerror(errno));

    // Check if the file immediately starts with the index, instead of vectors
    result.config.exclude_vectors = true;
    if (std::memcmp(result.head_buffer, default_magic(), std::strlen(default_magic())) == 0)
        return result;

    if (std::fseek(file.get(), 0L, SEEK_END) != 0)
        return result.failed("Can't infer file size");

    // Check if it starts with 32-bit
    std::size_t const file_size = std::ftell(file.get());

    std::uint32_t dimensions_u32[2]{0};
    std::memcpy(dimensions_u32, result.head_buffer, sizeof(dimensions_u32));
    std::size_t offset_if_u32 = std::size_t(dimensions_u32[0]) * dimensions_u32[1] + sizeof(dimensions_u32);

    std::uint64_t dimensions_u64[2]{0};
    std::memcpy(dimensions_u64, result.head_buffer, sizeof(dimensions_u64));
    std::size_t offset_if_u64 = std::size_t(dimensions_u64[0]) * dimensions_u64[1] + sizeof(dimensions_u64);

    // Check if it starts with 32-bit
    if (offset_if_u32 + sizeof(index_dense_head_buffer_t) < file_size) {
        if (std::fseek(file.get(), offset_if_u32, SEEK_SET) != 0)
            return result.failed(std::strerror(errno));
        read = std::fread(result.head_buffer, sizeof(index_dense_head_buffer_t), 1, file.get());
        if (!read)
            return result.failed(std::feof(file.get()) ? "End of file reached!" : std::strerror(errno));

        result.config.exclude_vectors = false;
        result.config.use_64_bit_dimensions = false;
        if (std::memcmp(result.head_buffer, default_magic(), std::strlen(default_magic())) == 0)
            return result;
    }

    // Check if it starts with 64-bit
    if (offset_if_u64 + sizeof(index_dense_head_buffer_t) < file_size) {
        if (std::fseek(file.get(), offset_if_u64, SEEK_SET) != 0)
            return result.failed(std::strerror(errno));
        read = std::fread(result.head_buffer, sizeof(index_dense_head_buffer_t), 1, file.get());
        if (!read)
            return result.failed(std::feof(file.get()) ? "End of file reached!" : std::strerror(errno));

        // Check if it starts with 64-bit
        result.config.exclude_vectors = false;
        result.config.use_64_bit_dimensions = true;
        if (std::memcmp(result.head_buffer, default_magic(), std::strlen(default_magic())) == 0)
            return result;
    }

    return result.failed("Not a dense USearch index!");
}

/**
 *  @brief  Oversimplified type-punned index for equidimensional vectors
 *          with automatic @b down-casting, hardware-specific @b SIMD metrics,
 *          and ability to @b remove existing vectors, common in Semantic Caching
 *          applications.
 *
 *  @section Serialization
 *
 *  The serialized binary form of `index_dense_gt` is made up of three parts:
 *      1. Binary matrix, aka the `.bbin` part,
 *      2. Metadata about used metrics, number of used vs free slots,
 *      3. The HNSW index in a binary form.
 *  The first (1.) generally starts with 2 integers - number of rows (vectors) and @b single-byte columns.
 *  The second (2.) starts with @b "usearch"-magic-string, used to infer the file type on open.
 *  The third (3.) is implemented by the underlying `index_gt` class.
 */
template <typename key_at = default_key_t, typename compressed_slot_at = default_slot_t> //
class index_dense_gt {
  public:
    using key_t = key_at;
    using compressed_slot_t = compressed_slot_at;
    using distance_t = distance_punned_t;
    using metric_t = metric_punned_t;

    using member_ref_t = member_ref_gt<key_t>;
    using member_cref_t = member_cref_gt<key_t>;

    using head_t = index_dense_head_t;
    using head_buffer_t = index_dense_head_buffer_t;
    using head_result_t = index_dense_head_result_t;

    using serialization_config_t = index_dense_serialization_config_t;

  private:
    /// @brief Schema: input buffer, bytes in input buffer, output buffer.
    using cast_t = std::function<bool(byte_t const*, std::size_t, byte_t*)>;
    /// @brief Punned index.
    using index_t = index_gt<                 //
        distance_t, key_t, compressed_slot_t, //
        aligned_allocator_gt<byte_t, 64>, memory_mapping_allocator_gt<64>>;
    using index_allocator_t = aligned_allocator_gt<index_t, 64>;

    using member_iterator_t = typename index_t::member_iterator_t;
    using member_citerator_t = typename index_t::member_citerator_t;

    /// @brief Punned metric object.
    class metric_proxy_t {
        index_dense_gt const* index_ = nullptr;

      public:
        metric_proxy_t(index_dense_gt const& index) noexcept : index_(&index) {}

        inline distance_t operator()(byte_t const* a, member_cref_t b) const noexcept { return f(a, v(b)); }
        inline distance_t operator()(member_cref_t a, member_cref_t b) const noexcept { return f(v(a), v(b)); }

        inline distance_t operator()(byte_t const* a, member_citerator_t b) const noexcept { return f(a, v(b)); }
        inline distance_t operator()(member_citerator_t a, member_citerator_t b) const noexcept {
            return f(v(a), v(b));
        }

        inline distance_t operator()(byte_t const* a, byte_t const* b) const noexcept { return f(a, b); }

        inline byte_t const* v(member_cref_t m) const noexcept { return index_->vectors_lookup_[get_slot(m)]; }
        inline byte_t const* v(member_citerator_t m) const noexcept { return index_->vectors_lookup_[get_slot(m)]; }
        inline distance_t f(byte_t const* a, byte_t const* b) const noexcept { return index_->metric_(a, b); }
    };

    index_dense_config_t config_;
    index_t* typed_ = nullptr;

    mutable std::vector<byte_t> cast_buffer_;
    struct casts_t {
        cast_t from_b1x8;
        cast_t from_i8;
        cast_t from_f16;
        cast_t from_f32;
        cast_t from_f64;

        cast_t to_b1x8;
        cast_t to_i8;
        cast_t to_f16;
        cast_t to_f32;
        cast_t to_f64;
    } casts_;

    /// @brief An instance of a potentially stateful `metric_t` used to initialize copies and forks.
    metric_t metric_;

    using vectors_tape_allocator_t = memory_mapping_allocator_gt<8>;
    /// @brief Allocator for the copied vectors, aligned to widest double-precision scalars.
    vectors_tape_allocator_t vectors_tape_allocator_;

    /// @brief For every managed `compressed_slot_t` stores a pointer to the allocated vector copy.
    mutable std::vector<byte_t*> vectors_lookup_;

    /// @brief Originally forms and array of integers [0, threads], marking all
    mutable std::vector<std::size_t> available_threads_;

    /// @brief Mutex, controlling concurrent access to `available_threads_`.
    mutable std::mutex available_threads_mutex_;

    using shared_mutex_t = std::mutex; // TODO: Find an OS-compatible solution
    using shared_lock_t = std::unique_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    struct key_and_slot_t {
        key_t key;
        compressed_slot_t slot;
    };

    struct lookup_key_hash_t {
        using is_transparent = void;
        std::size_t operator()(key_and_slot_t const& k) const noexcept { return std::hash<key_t>{}(k.key); }
        std::size_t operator()(key_t const& k) const noexcept { return std::hash<key_t>{}(k); }
    };

    struct lookup_key_same_t {
        using is_transparent = void;
        bool operator()(key_and_slot_t const& a, key_t const& b) const noexcept { return a.key == b; }
        bool operator()(key_t const& a, key_and_slot_t const& b) const noexcept { return a == b.key; }
        bool operator()(key_and_slot_t const& a, key_and_slot_t const& b) const noexcept {
            return a.key == b.key && a.slot == b.slot;
        }
    };

    /// @brief Multi-Map from keys to IDs, and allocated vectors.
    tsl::robin_set<key_and_slot_t, lookup_key_hash_t, lookup_key_same_t> slot_lookup_;

    /// @brief Mutex, controlling concurrent access to `slot_lookup_`.
    mutable shared_mutex_t slot_lookup_mutex_;

    /// @brief Ring-shaped queue of deleted entries, to be reused on future insertions.
    ring_gt<compressed_slot_t> free_keys_;

    /// @brief Mutex, controlling concurrent access to `free_keys_`.
    mutable std::mutex free_keys_mutex_;

    /// @brief A constant for the reserved key value, used to mark deleted entries.
    key_t free_key_ = default_free_value<key_t>();

  public:
    using search_result_t = typename index_t::search_result_t;
    using add_result_t = typename index_t::add_result_t;
    using stats_t = typename index_t::stats_t;
    using match_t = typename index_t::match_t;

    index_dense_gt() = default;
    index_dense_gt(index_dense_gt&& other)
        : config_(std::move(other.config_)),

          typed_(exchange(other.typed_, nullptr)),     //
          cast_buffer_(std::move(other.cast_buffer_)), //
          casts_(std::move(other.casts_)),             //
          metric_(std::move(other.metric_)),           //

          vectors_tape_allocator_(std::move(other.vectors_tape_allocator_)), //
          vectors_lookup_(std::move(other.vectors_lookup_)),                 //

          available_threads_(std::move(other.available_threads_)), //
          slot_lookup_(std::move(other.slot_lookup_)),             //
          free_keys_(std::move(other.free_keys_)),                 //
          free_key_(std::move(other.free_key_)) {}                 //

    index_dense_gt& operator=(index_dense_gt&& other) {
        swap(other);
        return *this;
    }

    /**
     *  @brief Swaps the contents of this index with another index.
     *  @param other The other index to swap with.
     */
    void swap(index_dense_gt& other) {
        std::swap(config_, other.config_);

        std::swap(typed_, other.typed_);
        std::swap(cast_buffer_, other.cast_buffer_);
        std::swap(casts_, other.casts_);
        std::swap(metric_, other.metric_);

        std::swap(vectors_tape_allocator_, other.vectors_tape_allocator_);
        std::swap(vectors_lookup_, other.vectors_lookup_);

        std::swap(available_threads_, other.available_threads_);
        std::swap(slot_lookup_, other.slot_lookup_);
        std::swap(free_keys_, other.free_keys_);
        std::swap(free_key_, other.free_key_);
    }

    ~index_dense_gt() {
        if (typed_)
            typed_->~index_t();
        index_allocator_t{}.deallocate(typed_, 1);
        typed_ = nullptr;
    }

    /**
     *  @brief Constructs an instance of ::index_dense_gt.
     *  @param[in] metric One of the provided or an @b ad-hoc metric, type-punned.
     *  @param[in] config The index configuration (optional).
     *  @param[in] free_label The key used for freed vectors (optional).
     *  @return An instance of ::index_dense_gt.
     */
    static index_dense_gt make(           //
        metric_t metric,                  //
        index_dense_config_t config = {}, //
        key_t free_label = default_free_value<key_t>()) {

        scalar_kind_t scalar_kind = metric.scalar_kind();
        std::size_t hardware_threads = std::thread::hardware_concurrency();

        index_dense_gt result;
        result.config_ = config;
        result.cast_buffer_.resize(hardware_threads * metric.bytes_per_vector());
        result.casts_ = make_casts_(scalar_kind);
        result.metric_ = metric;
        result.free_key_ = free_label;

        // Fill the thread IDs.
        result.available_threads_.resize(hardware_threads);
        std::iota(result.available_threads_.begin(), result.available_threads_.end(), 0ul);

        // Available since C11, but only C++17, so we use the C version.
        index_t* raw = index_allocator_t{}.allocate(1);
        new (raw) index_t(config);
        result.typed_ = raw;
        return result;
    }

    static index_dense_gt make(char const* path, bool view = false) {
        index_dense_metadata_result_t meta = index_dense_metadata(path);
        if (!meta)
            return {};
        metric_punned_t metric(meta.head.dimensions, meta.head.kind_metric, meta.head.kind_scalar);
        index_dense_gt result = make(metric);
        if (!result)
            return result;
        if (view)
            result.view(path);
        else
            result.load(path);
        return result;
    }

    explicit operator bool() const { return typed_; }
    std::size_t connectivity() const { return typed_->connectivity(); }
    std::size_t size() const { return typed_->size() - free_keys_.size(); }
    std::size_t capacity() const { return typed_->capacity(); }
    std::size_t max_level() const noexcept { return typed_->max_level(); }
    index_dense_config_t const& config() const { return config_; }
    index_limits_t const& limits() const { return typed_->limits(); }

    // The metric and its properties
    metric_t const& metric() const { return metric_; }
    scalar_kind_t scalar_kind() const noexcept { return metric_.scalar_kind(); }
    std::size_t bytes_per_vector() const noexcept { return metric_.bytes_per_vector(); }
    std::size_t scalar_words() const noexcept { return metric_.scalar_words(); }
    std::size_t dimensions() const noexcept { return metric_.dimensions(); }

    // Fetching and changing search criteria
    std::size_t expansion_add() const { return config_.expansion_add; }
    std::size_t expansion_search() const { return config_.expansion_search; }
    void change_expansion_add(std::size_t n) { config_.expansion_add = n; }
    void change_expansion_search(std::size_t n) { config_.expansion_search = n; }

    member_citerator_t cbegin() const { return typed_->cbegin(); }
    member_citerator_t cend() const { return typed_->cend(); }
    member_citerator_t begin() const { return typed_->begin(); }
    member_citerator_t end() const { return typed_->end(); }
    member_iterator_t begin() { return typed_->begin(); }
    member_iterator_t end() { return typed_->end(); }

    stats_t stats() const { return typed_->stats(); }
    stats_t stats(std::size_t level) const { return typed_->stats(level); }

    /**
     *  @brief  A relatively accurate lower bound on the amount of memory consumed by the system.
     *          In practice it's error will be below 10%.
     *
     *  @see    `stream_length` for the length of the binary serialized representation.
     */
    std::size_t memory_usage() const {
        return                                          //
            typed_->memory_usage(0) +                   //
            typed_->tape_allocator().total_wasted() +   //
            typed_->tape_allocator().total_reserved() + //
            vectors_tape_allocator_.total_allocated();
    }

    // clang-format off
    add_result_t add(key_t key, b1x8_t const* vector) { return add_(key, vector, casts_.from_b1x8); }
    add_result_t add(key_t key, i8_bits_t const* vector) { return add_(key, vector, casts_.from_i8); }
    add_result_t add(key_t key, f16_t const* vector) { return add_(key, vector, casts_.from_f16); }
    add_result_t add(key_t key, f32_t const* vector) { return add_(key, vector, casts_.from_f32); }
    add_result_t add(key_t key, f64_t const* vector) { return add_(key, vector, casts_.from_f64); }

    add_result_t add(key_t key, b1x8_t const* vector, index_dense_update_config_t config) { return add_(key, vector, config, casts_.from_b1x8); }
    add_result_t add(key_t key, i8_bits_t const* vector, index_dense_update_config_t config) { return add_(key, vector, config, casts_.from_i8); }
    add_result_t add(key_t key, f16_t const* vector, index_dense_update_config_t config) { return add_(key, vector, config, casts_.from_f16); }
    add_result_t add(key_t key, f32_t const* vector, index_dense_update_config_t config) { return add_(key, vector, config, casts_.from_f32); }
    add_result_t add(key_t key, f64_t const* vector, index_dense_update_config_t config) { return add_(key, vector, config, casts_.from_f64); }

    search_result_t search(b1x8_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_b1x8); }
    search_result_t search(i8_bits_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_i8); }
    search_result_t search(f16_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f16); }
    search_result_t search(f32_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f32); }
    search_result_t search(f64_t const* vector, std::size_t wanted) const { return search_(vector, wanted, casts_.from_f64); }

    search_result_t search(b1x8_t const* vector, std::size_t wanted, index_search_config_t config) const { return search_(vector, wanted, config, casts_.from_b1x8); }
    search_result_t search(i8_bits_t const* vector, std::size_t wanted, index_search_config_t config) const { return search_(vector, wanted, config, casts_.from_i8); }
    search_result_t search(f16_t const* vector, std::size_t wanted, index_search_config_t config) const { return search_(vector, wanted, config, casts_.from_f16); }
    search_result_t search(f32_t const* vector, std::size_t wanted, index_search_config_t config) const { return search_(vector, wanted, config, casts_.from_f32); }
    search_result_t search(f64_t const* vector, std::size_t wanted, index_search_config_t config) const { return search_(vector, wanted, config, casts_.from_f64); }

    bool get(key_t key, b1x8_t* vector) const { return get_(key, vector, casts_.to_b1x8); }
    bool get(key_t key, i8_bits_t* vector) const { return get_(key, vector, casts_.to_i8); }
    bool get(key_t key, f16_t* vector) const { return get_(key, vector, casts_.to_f16); }
    bool get(key_t key, f32_t* vector) const { return get_(key, vector, casts_.to_f32); }
    bool get(key_t key, f64_t* vector) const { return get_(key, vector, casts_.to_f64); }
    // clang-format on

    /**
     *  @brief Reserves memory for the index and the labeled lookup.
     *  @return `true` if the memory reservation was successful, `false` otherwise.
     */
    bool reserve(index_limits_t limits) {
        {
            unique_lock_t lock(slot_lookup_mutex_);
            slot_lookup_.reserve(limits.members);
            vectors_lookup_.resize(limits.members);
        }
        return typed_->reserve(limits);
    }

    /**
     *  @brief Erases all the vectors from the index.
     *
     *  Will change `size()` to zero, but will keep the same `capacity()`.
     *  Will keep the number of available threads/contexts the same as it was.
     */
    void clear() {
        unique_lock_t lookup_lock(slot_lookup_mutex_);

        std::unique_lock<std::mutex> free_lock(free_keys_mutex_);
        typed_->clear();
        slot_lookup_.clear();
        vectors_lookup_.clear();
        free_keys_.clear();
        vectors_tape_allocator_.reset();
    }

    /**
     *  @brief Erases all members from index, closing files, and returning RAM to OS.
     *
     *  Will change both `size()` and `capacity()` to zero.
     *  Will deallocate all threads/contexts.
     *  If the index is memory-mapped - releases the mapping and the descriptor.
     */
    void reset() {
        unique_lock_t lookup_lock(slot_lookup_mutex_);

        std::unique_lock<std::mutex> free_lock(free_keys_mutex_);
        std::unique_lock<std::mutex> available_threads_lock(available_threads_mutex_);
        typed_->reset();
        slot_lookup_.clear();
        vectors_lookup_.clear();
        free_keys_.clear();
        vectors_tape_allocator_.reset();
        available_threads_.clear();
    }

    /**
     *  @brief Saves the index to a file.
     *  @param[in] path The path to the file.
     *  @param[in] config Configuration parameters for exports.
     *  @return Outcome descriptor explicitly convertible to boolean.
     */
    serialization_result_t save(output_file_t file, serialization_config_t config = {}) const {
        serialization_result_t result = file.open_if_not();
        if (result)
            stream(
                [&](void* buffer, std::size_t length) {
                    result = file.write(buffer, length);
                    return !!result;
                },
                config);
        return result;
    }

    /**
     *  @brief  Saves serialized binary index representation to a stream.
     */
    template <typename output_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t stream(output_callback_at&& callback, serialization_config_t config = {}) const noexcept {

        serialization_result_t result;
        std::uint64_t matrix_rows = 0;
        std::uint64_t matrix_cols = 0;

        // We may not want to put the vectors into the same file
        if (!config.exclude_vectors) {
            // Save the matrix size
            if (!config.use_64_bit_dimensions) {
                std::uint32_t dimensions[2];
                dimensions[0] = static_cast<std::uint32_t>(typed_->size());
                dimensions[1] = static_cast<std::uint32_t>(metric_.bytes_per_vector());
                if (!callback(&dimensions, sizeof(dimensions)))
                    return result.failed("Failed to serialize into stream");
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            } else {
                std::uint64_t dimensions[2];
                dimensions[0] = static_cast<std::uint64_t>(typed_->size());
                dimensions[1] = static_cast<std::uint64_t>(metric_.bytes_per_vector());
                if (!callback(&dimensions, sizeof(dimensions)))
                    return result.failed("Failed to serialize into stream");
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            }

            // Dump the vectors one after another
            for (std::uint64_t i = 0; i != matrix_rows; ++i) {
                byte_t* vector = vectors_lookup_[i];
                if (!callback(vector, matrix_cols))
                    return result.failed("Failed to serialize into stream");
            }
        }

        // Augment metadata
        {
            index_dense_head_buffer_t buffer;
            std::memset(buffer, 0, sizeof(buffer));
            index_dense_head_t head{buffer};
            std::memcpy(buffer, default_magic(), std::strlen(default_magic()));

            // Describe software version
            using version_t = index_dense_head_t::version_t;
            head.version_major = static_cast<version_t>(USEARCH_VERSION_MAJOR);
            head.version_minor = static_cast<version_t>(USEARCH_VERSION_MINOR);
            head.version_patch = static_cast<version_t>(USEARCH_VERSION_PATCH);

            // Describes types used
            head.kind_metric = metric_.metric_kind();
            head.kind_scalar = metric_.scalar_kind();
            head.kind_key = unum::usearch::scalar_kind<key_t>();
            head.kind_compressed_slot = unum::usearch::scalar_kind<compressed_slot_t>();

            head.count_present = size();
            head.count_deleted = typed_->size() - size();
            head.dimensions = dimensions();

            if (!callback(&buffer, sizeof(buffer)))
                return result.failed("Failed to serialize into stream");
        }

        // Save the actual proximity graph
        return typed_->stream(std::forward<output_callback_at>(callback));
    }

    /**
     *  @brief  Estimate the binary length (in bytes) of the serialized index.
     */
    std::size_t stream_length(serialization_config_t config = {}) const noexcept {
        std::size_t dimensions_length = 0;
        std::size_t matrix_length = 0;
        if (!config.exclude_vectors) {
            dimensions_length = config.use_64_bit_dimensions ? sizeof(std::uint64_t) * 2 : sizeof(std::uint32_t) * 2;
            matrix_length = typed_->size() * metric_.bytes_per_vector();
        }
        return dimensions_length + matrix_length + sizeof(index_dense_head_buffer_t) + typed_->stream_length();
    }

    /**
     *  @brief Parses the index from file to RAM.
     *  @param[in] path The path to the file.
     *  @param[in] config Configuration parameters for imports.
     *  @return Outcome descriptor explicitly convertible to boolean.
     */
    serialization_result_t load(input_file_t file, serialization_config_t config = {}) {

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        std::uint64_t matrix_rows = 0;
        std::uint64_t matrix_cols = 0;

        // We may not want to load the vectors from the same file, or allow attaching them afterwards
        if (!config.exclude_vectors) {
            // Save the matrix size
            if (!config.use_64_bit_dimensions) {
                std::uint32_t dimensions[2];
                result = file.read(&dimensions, sizeof(dimensions));
                if (!result)
                    return result;
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            } else {
                std::uint64_t dimensions[2];
                result = file.read(&dimensions, sizeof(dimensions));
                if (!result)
                    return result;
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            }
            // Load the vectors one after another
            vectors_lookup_.resize(matrix_rows);
            for (std::uint64_t slot = 0; slot != matrix_rows; ++slot) {
                byte_t* vector = vectors_tape_allocator_.allocate(matrix_cols);
                result = file.read(vector, matrix_cols);
                if (!result)
                    return result;
                vectors_lookup_[slot] = vector;
            }
        }

        // Load metadata and choose the right metric
        {
            index_dense_head_buffer_t buffer;
            result = file.read(buffer, sizeof(buffer));
            if (!result)
                return result;

            index_dense_head_t head{buffer};
            if (std::memcmp(buffer, default_magic(), std::strlen(default_magic())) != 0)
                return result.failed("Magic header mismatch - the file isn't an index");

            // Validate the software version
            if (head.version_major != USEARCH_VERSION_MAJOR)
                return result.failed("File format may be different, please rebuild");

            // Check the types used
            if (head.kind_key != unum::usearch::scalar_kind<key_t>())
                return result.failed("Key type doesn't match, consider rebuilding");
            if (head.kind_compressed_slot != unum::usearch::scalar_kind<compressed_slot_t>())
                return result.failed("Slot type doesn't match, consider rebuilding");

            metric_ = metric_t(head.dimensions, head.kind_metric, head.kind_scalar);
        }

        // Pull the actual proximity graph
        result = typed_->load(std::move(file));
        if (!result)
            return result;
        if (typed_->size() != static_cast<std::size_t>(matrix_rows))
            return result.failed("Index size and the number of vectors doesn't match");

        reindex_labels_();
        return result;
    }

    /**
     *  @brief Parses the index from file, without loading it into RAM.
     *  @param[in] path The path to the file.
     *  @param[in] config Configuration parameters for imports.
     *  @return Outcome descriptor explicitly convertible to boolean.
     */
    serialization_result_t view(memory_mapped_file_t file, std::size_t offset = 0, serialization_config_t config = {}) {

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        std::uint64_t matrix_rows = 0;
        std::uint64_t matrix_cols = 0;
        span_punned_t vectors_buffer;

        // We may not want to fetch the vectors from the same file, or allow attaching them afterwards
        if (!config.exclude_vectors) {
            // Save the matrix size
            if (!config.use_64_bit_dimensions) {
                std::uint32_t dimensions[2];
                if (file.size() - offset < sizeof(dimensions))
                    return result.failed("File is corrupted and lacks matrix dimensions");
                std::memcpy(&dimensions, file.data() + offset, sizeof(dimensions));
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
                offset += sizeof(dimensions);
            } else {
                std::uint64_t dimensions[2];
                if (file.size() - offset < sizeof(dimensions))
                    return result.failed("File is corrupted and lacks matrix dimensions");
                std::memcpy(&dimensions, file.data() + offset, sizeof(dimensions));
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
                offset += sizeof(dimensions);
            }
            vectors_buffer = {file.data() + offset, static_cast<std::size_t>(matrix_rows * matrix_cols)};
            offset += vectors_buffer.size();
        }

        // Load metadata and choose the right metric
        {
            index_dense_head_buffer_t buffer;
            if (file.size() - offset < sizeof(buffer))
                return result.failed("File is corrupted and lacks a header");

            std::memcpy(buffer, file.data() + offset, sizeof(buffer));

            index_dense_head_t head{buffer};
            if (std::memcmp(buffer, default_magic(), std::strlen(default_magic())) != 0)
                return result.failed("Magic header mismatch - the file isn't an index");

            // Validate the software version
            if (head.version_major != USEARCH_VERSION_MAJOR)
                return result.failed("File format may be different, please rebuild");

            // Check the types used
            if (head.kind_key != unum::usearch::scalar_kind<key_t>())
                return result.failed("Key type doesn't match, consider rebuilding");
            if (head.kind_compressed_slot != unum::usearch::scalar_kind<compressed_slot_t>())
                return result.failed("Slot type doesn't match, consider rebuilding");

            metric_ = metric_t(head.dimensions, head.kind_metric, head.kind_scalar);
            offset += sizeof(buffer);
        }

        // Pull the actual proximity graph
        result = typed_->view(std::move(file), offset);
        if (!result)
            return result;
        if (typed_->size() != static_cast<std::size_t>(matrix_rows))
            return result.failed("Index size and the number of vectors doesn't match");

        // Address the vectors
        vectors_lookup_.resize(matrix_rows);
        if (!config.exclude_vectors)
            for (std::uint64_t slot = 0; slot != matrix_rows; ++slot)
                vectors_lookup_[slot] = (byte_t*)vectors_buffer.data() + matrix_cols * slot;

        reindex_labels_();
        return result;
    }

    /**
     *  @brief Checks if a vector with specified key is present.
     *  @return `true` if the key is present in the index, `false` otherwise.
     */
    bool contains(key_t key) const {
        shared_lock_t lock(slot_lookup_mutex_);
        return slot_lookup_.contains(key);
    }

    /**
     *  @brief Count the number of vectors with specified key present.
     *  @return Zero if nothing is found, a positive integer otherwise.
     */
    std::size_t count(key_t key) const {
        shared_lock_t lock(slot_lookup_mutex_);
        return slot_lookup_.count(key);
    }

    struct labeling_result_t {
        error_t error{};
        std::size_t completed{};

        explicit operator bool() const noexcept { return !error; }
        labeling_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /**
     *  @brief Removes an entry with the specified key from the index.
     *  @param[in] key The key of the entry to remove.
     *  @return The ::labeling_result_t indicating the result of the removal operation.
     *          If the removal was successful, `result.completed` will be `true`.
     *          If the key was not found in the index, `result.completed` will be `false`.
     *          If an error occurred during the removal operation, `result.error` will contain an error message.
     */
    labeling_result_t remove(key_t key) {
        labeling_result_t result;

        unique_lock_t lookup_lock(slot_lookup_mutex_);
        auto labeled_iterator = slot_lookup_.find(key);
        if (labeled_iterator == slot_lookup_.end())
            return result;

        // Grow the removed entries ring, if needed
        std::unique_lock<std::mutex> free_lock(free_keys_mutex_);
        if (free_keys_.size() == free_keys_.capacity())
            if (!free_keys_.reserve((std::max<std::size_t>)(free_keys_.capacity() * 2, 64ul)))
                return result.failed("Can't allocate memory for a free-list");

        // A removed entry would be:
        // - present in `free_keys_`
        // - missing in the `slot_lookup_`
        // - marked in the `typed_` index with a `free_key_`
        compressed_slot_t slot = (*labeled_iterator).slot;
        free_keys_.push(slot);
        slot_lookup_.erase(labeled_iterator);
        typed_->at(slot).key = free_key_;
        result.completed = true;

        return result;
    }

    /**
     *  @brief Removes multiple entries with the specified keys from the index.
     *  @param[in] labels_begin The beginning of the keys range.
     *  @param[in] labels_end The ending of the keys range.
     *  @return The ::labeling_result_t indicating the result of the removal operation.
     *          `result.completed` will contain the number of keys that were successfully removed.
     *          `result.error` will contain an error message if an error occurred during the removal operation.
     */
    template <typename labels_iterator_at>
    labeling_result_t remove(labels_iterator_at&& labels_begin, labels_iterator_at&& labels_end) {

        labeling_result_t result;
        unique_lock_t lookup_lock(slot_lookup_mutex_);
        std::unique_lock<std::mutex> free_lock(free_keys_mutex_);

        // Grow the removed entries ring, if needed
        std::size_t count_requests = std::distance(labels_begin, labels_end);
        if (!free_keys_.reserve(free_keys_.size() + count_requests))
            return result.failed("Can't allocate memory for a free-list");

        // Remove them one-by-one
        for (auto label_it = labels_begin; label_it != labels_end; ++label_it) {
            key_t key = *label_it;
            auto labeled_iterator = slot_lookup_.find(key);
            if (labeled_iterator == slot_lookup_.end())
                continue;

            // A removed entry would be:
            // - present in `free_keys_`
            // - missing in the `slot_lookup_`
            // - marked in the `typed_` index with a `free_key_`
            compressed_slot_t slot = (*labeled_iterator).slot;
            free_keys_.push(slot);
            slot_lookup_.erase(labeled_iterator);
            typed_->at(slot).key = free_key_;
            result.completed += 1;
        }

        return result;
    }

    /**
     *  @brief Renames an entry with the specified key to a new key.
     *  @param[in] from The current key of the entry to rename.
     *  @param[in] to The new key to assign to the entry.
     *  @return The ::labeling_result_t indicating the result of the rename operation.
     *          If the rename was successful, `result.completed` will be `true`.
     *          If the entry with the current key was not found, `result.completed` will be `false`.
     */
    labeling_result_t rename(key_t from, key_t to) {
        labeling_result_t result;
        unique_lock_t lookup_lock(slot_lookup_mutex_);
        auto labeled_iterator = slot_lookup_.find(from);
        if (labeled_iterator == slot_lookup_.end())
            return result;

        compressed_slot_t slot = (*labeled_iterator).slot;
        key_and_slot_t key_and_slot{to, slot};
        slot_lookup_.erase(labeled_iterator);
        slot_lookup_.insert(key_and_slot);
        typed_->at(slot).key = to;
        result.completed = true;
        return result;
    }

    /**
     *  @brief Exports a range of keys for the vectors present in the index.
     *  @param[out] keys Pointer to the array where the keys will be exported.
     *  @param[in] offset The number of keys to skip. Useful for pagination.
     *  @param[in] limit The maximum number of keys to export, that can fit in ::keys.
     */
    void export_keys(key_t* keys, std::size_t offset, std::size_t limit) const {
        shared_lock_t lock(slot_lookup_mutex_);
        auto it = slot_lookup_.begin();
        offset = (std::min)(offset, slot_lookup_.size());
        std::advance(it, offset);
        for (; it != slot_lookup_.end() && limit; ++it, ++keys, --limit)
            *keys = (*it).key;
    }

    struct copy_result_t {
        index_dense_gt index;
        error_t error;

        explicit operator bool() const noexcept { return !error; }
        copy_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /**
     *  @brief Copies the ::index_dense_gt @b with all the data in it.
     *  @param config The copy configuration (optional).
     *  @return A copy of the ::index_dense_gt instance.
     */
    copy_result_t copy(index_dense_copy_config_t config = {}) const {
        copy_result_t result = fork();
        if (!result)
            return result;

        auto typed_result = typed_->copy(config);
        if (!typed_result)
            return result.failed(std::move(typed_result.error));

        // Export the free (removed) slot numbers
        index_dense_gt& copy = result.index;
        if (!copy.free_keys_.reserve(free_keys_.size()))
            return result.failed(std::move(typed_result.error));
        for (std::size_t i = 0; i != free_keys_.size(); ++i)
            copy.free_keys_.push(free_keys_[i]);

        // Allocate buffers and move the vectors themselves
        if (!config.force_vector_copy && copy.config_.exclude_vectors)
            copy.vectors_lookup_ = vectors_lookup_;
        else {
            copy.vectors_lookup_.resize(vectors_lookup_.size());
            for (std::size_t slot = 0; slot != vectors_lookup_.size(); ++slot)
                copy.vectors_lookup_[slot] = copy.vectors_tape_allocator_.allocate(copy.metric_.bytes_per_vector());
            if (std::count(copy.vectors_lookup_.begin(), copy.vectors_lookup_.end(), nullptr))
                return result.failed("Out of memory!");
            for (std::size_t slot = 0; slot != vectors_lookup_.size(); ++slot)
                std::memcpy(copy.vectors_lookup_[slot], vectors_lookup_[slot], metric_.bytes_per_vector());
        }

        copy.slot_lookup_ = slot_lookup_;
        *copy.typed_ = std::move(typed_result.index);
        return result;
    }

    /**
     *  @brief Copies the ::index_dense_gt model @b without any data.
     *  @return A similarly configured ::index_dense_gt instance.
     */
    copy_result_t fork() const {
        copy_result_t result;
        index_dense_gt& other = result.index;

        other.config_ = config_;
        other.cast_buffer_ = cast_buffer_;
        other.casts_ = casts_;

        other.metric_ = metric_;
        other.available_threads_ = available_threads_;
        other.free_key_ = free_key_;

        index_t* raw = index_allocator_t{}.allocate(1);
        if (!raw)
            return result.failed("Can't allocate the index");

        new (raw) index_t(config());
        other.typed_ = raw;
        return result;
    }

    struct compaction_result_t {
        error_t error{};
        std::size_t pruned_edges{};

        explicit operator bool() const noexcept { return !error; }
        compaction_result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

    /**
     *  @brief Performs compaction on the index, pruning links to removed entries.
     *  @param executor The executor parallel processing. Default ::dummy_executor_t single-threaded.
     *  @param progress The progress tracker instance to use. Default ::dummy_progress_t reports nothing.
     *  @return The ::compaction_result_t indicating the result of the compaction operation.
     *          `result.pruned_edges` will contain the number of edges that were removed.
     *          `result.error` will contain an error message if an error occurred during the compaction operation.
     */
    template <typename executor_at = dummy_executor_t, typename progress_at = dummy_progress_t>
    compaction_result_t isolate(executor_at&& executor = executor_at{}, progress_at&& progress = progress_at{}) {
        compaction_result_t result;
        std::atomic<std::size_t> pruned_edges;
        auto disallow = [&](member_cref_t const& member) noexcept {
            bool freed = member.key == free_key_;
            pruned_edges += freed;
            return freed;
        };
        typed_->isolate(disallow, std::forward<executor_at>(executor), std::forward<progress_at>(progress));
        result.pruned_edges = pruned_edges;
        return result;
    }

    class values_proxy_t {
        index_dense_gt const* index_;

      public:
        values_proxy_t(index_dense_gt const& index) noexcept : index_(&index) {}
        byte_t const* operator[](compressed_slot_t slot) const noexcept { return index_->vectors_lookup_[slot]; }
        byte_t const* operator[](member_citerator_t it) const noexcept { return index_->vectors_lookup_[get_slot(it)]; }
    };

    /**
     *  @brief Performs compaction on the index, pruning links to removed entries.
     *  @param executor The executor parallel processing. Default ::dummy_executor_t single-threaded.
     *  @param progress The progress tracker instance to use. Default ::dummy_progress_t reports nothing.
     *  @return The ::compaction_result_t indicating the result of the compaction operation.
     *          `result.pruned_edges` will contain the number of edges that were removed.
     *          `result.error` will contain an error message if an error occurred during the compaction operation.
     */
    template <typename executor_at = dummy_executor_t, typename progress_at = dummy_progress_t>
    compaction_result_t compact(executor_at&& executor = executor_at{}, progress_at&& progress = progress_at{}) {
        compaction_result_t result;

        std::vector<byte_t*> new_vectors_lookup(vectors_lookup_.size());
        vectors_tape_allocator_t new_vectors_allocator;

        auto track_slot_change = [&](key_t, compressed_slot_t old_slot, compressed_slot_t new_slot) {
            byte_t* new_vector = new_vectors_allocator.allocate(metric_.bytes_per_vector());
            byte_t* old_vector = vectors_lookup_[old_slot];
            std::memcpy(new_vector, old_vector, metric_.bytes_per_vector());
            new_vectors_lookup[new_slot] = new_vector;
        };
        typed_->compact(values_proxy_t{*this}, metric_proxy_t{*this}, track_slot_change,
                        std::forward<executor_at>(executor), std::forward<progress_at>(progress));
        vectors_lookup_ = std::move(new_vectors_lookup);
        vectors_tape_allocator_ = std::move(new_vectors_allocator);
        return result;
    }

    template <                                                     //
        typename man_to_woman_at = dummy_label_to_label_mapping_t, //
        typename woman_to_man_at = dummy_label_to_label_mapping_t, //
        typename executor_at = dummy_executor_t,                   //
        typename progress_at = dummy_progress_t                    //
        >
    join_result_t join(                                     //
        index_dense_gt const& women,                        //
        index_join_config_t config = {},                    //
        man_to_woman_at&& man_to_woman = man_to_woman_at{}, //
        woman_to_man_at&& woman_to_man = woman_to_man_at{}, //
        executor_at&& executor = executor_at{},             //
        progress_at&& progress = progress_at{}) const {

        index_dense_gt const& men = *this;
        return unum::usearch::join(                      //
            *men.typed_, *women.typed_,                  //
            values_proxy_t{men}, values_proxy_t{women},  //
            metric_proxy_t{men}, metric_proxy_t{women},  //
            config,                                      //
            std::forward<man_to_woman_at>(man_to_woman), //
            std::forward<woman_to_man_at>(woman_to_man), //
            std::forward<executor_at>(executor),         //
            std::forward<progress_at>(progress));
    }

  private:
    struct thread_lock_t {
        index_dense_gt const& parent;
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
    add_result_t add_(key_t key, scalar_at const* vector, index_dense_update_config_t config, cast_t const& cast) {

        if (!config_.allow_key_collisions && contains(key))
            return add_result_t{}.failed("Duplicate keys not allowed in high-level wrappers");

        // Cast the vector, if needed for compatibility with `metric_`
        bool copy_vector = !config_.exclude_vectors || config.force_vector_copy;
        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        {
            byte_t* casted_data = cast_buffer_.data() + metric_.bytes_per_vector() * config.thread;
            bool casted = cast(vector_data, dimensions(), casted_data);
            if (casted)
                vector_data = casted_data, copy_vector = true;
        }

        // Check if there are some removed entries, whose nodes we can reuse
        compressed_slot_t free_slot = default_free_value<compressed_slot_t>();
        {
            std::unique_lock<std::mutex> lock(free_keys_mutex_);
            free_keys_.try_pop(free_slot);
        }

        // Perform the insertion or the update
        bool reuse_node = free_slot != default_free_value<compressed_slot_t>();
        auto on_success = [&](member_ref_t member) {
            unique_lock_t slot_lock(slot_lookup_mutex_);
            slot_lookup_.insert({key, static_cast<compressed_slot_t>(member.slot)});
            if (copy_vector) {
                if (!reuse_node)
                    vectors_lookup_[member.slot] = vectors_tape_allocator_.allocate(metric_.bytes_per_vector());
                std::memcpy(vectors_lookup_[member.slot], vector_data, metric_.bytes_per_vector());
            } else
                vectors_lookup_[member.slot] = (byte_t*)vector_data;
        };

        metric_proxy_t metric{*this};
        return reuse_node //
                   ? typed_->update(typed_->iterator_at(free_slot), key, vector_data, metric, config, on_success)
                   : typed_->add(key, vector_data, metric, config, on_success);
    }

    template <typename scalar_at>
    search_result_t search_(                         //
        scalar_at const* vector, std::size_t wanted, //
        index_search_config_t config, cast_t const& cast) const {

        // Cast the vector, if needed for compatibility with `metric_`
        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        {
            byte_t* casted_data = cast_buffer_.data() + metric_.bytes_per_vector() * config.thread;
            bool casted = cast(vector_data, dimensions(), casted_data);
            if (casted)
                vector_data = casted_data;
        }

        auto allow = [=](member_cref_t const& member) noexcept { return member.key != free_key_; };
        return typed_->search(vector_data, wanted, metric_proxy_t{*this}, config, allow);
    }

    compressed_slot_t lookup_id_(key_t key) const {
        shared_lock_t lock(slot_lookup_mutex_);
        return slot_lookup_.at(key);
    }

    void reindex_labels_() {

        // Estimate number of entries first
        std::size_t count_total = typed_->size();
        std::size_t count_removed = 0;
        for (std::size_t i = 0; i != count_total; ++i) {
            member_cref_t member = typed_->at(i);
            count_removed += member.key == free_key_;
        }

        // Pull entries from the underlying `typed_` into either
        // into `slot_lookup_`, or `free_keys_` if they are unused.
        unique_lock_t lock(slot_lookup_mutex_);
        slot_lookup_.clear();
        slot_lookup_.reserve(count_total - count_removed);
        free_keys_.clear();
        free_keys_.reserve(count_removed);
        for (std::size_t i = 0; i != typed_->size(); ++i) {
            member_cref_t member = typed_->at(i);
            if (member.key == free_key_)
                free_keys_.push(static_cast<compressed_slot_t>(i));
            else
                slot_lookup_.insert(key_and_slot_t{key_t(member.key), static_cast<compressed_slot_t>(i)});
        }
    }

    template <typename scalar_at> bool get_(key_t key, scalar_at* reconstructed, cast_t const& cast) const {
        compressed_slot_t slot;
        // Find the matching ID
        {
            shared_lock_t lock(slot_lookup_mutex_);
            auto it = slot_lookup_.find(key);
            if (it == slot_lookup_.end())
                return false;
            slot = (*it).slot;
        }
        // Export the entry
        byte_t const* punned_vector = reinterpret_cast<byte_t const*>(vectors_lookup_[slot]);
        bool casted = cast(punned_vector, dimensions(), (byte_t*)reconstructed);
        if (!casted)
            std::memcpy(reconstructed, punned_vector, metric_.bytes_per_vector());
        return true;
    }

    template <typename scalar_at> add_result_t add_(key_t key, scalar_at const* vector, cast_t const& cast) {
        thread_lock_t lock = thread_lock_();
        index_dense_update_config_t update_config;
        update_config.thread = lock.thread_id;
        update_config.expansion = config_.expansion_add;
        return add_(key, vector, update_config, cast);
    }

    template <typename scalar_at>
    search_result_t search_(                         //
        scalar_at const* vector, std::size_t wanted, //
        cast_t const& cast) const {
        thread_lock_t lock = thread_lock_();
        index_search_config_t search_config;
        search_config.thread = lock.thread_id;
        search_config.expansion = config_.expansion_search;
        return search_(vector, wanted, search_config, cast);
    }

    template <typename to_scalar_at> static casts_t make_casts_() {
        casts_t result;

        result.from_b1x8 = cast_gt<b1x8_t, to_scalar_at>{};
        result.from_i8 = cast_gt<i8_bits_t, to_scalar_at>{};
        result.from_f16 = cast_gt<f16_t, to_scalar_at>{};
        result.from_f32 = cast_gt<f32_t, to_scalar_at>{};
        result.from_f64 = cast_gt<f64_t, to_scalar_at>{};

        result.to_b1x8 = cast_gt<to_scalar_at, b1x8_t>{};
        result.to_i8 = cast_gt<to_scalar_at, i8_bits_t>{};
        result.to_f16 = cast_gt<to_scalar_at, f16_t>{};
        result.to_f32 = cast_gt<to_scalar_at, f32_t>{};
        result.to_f64 = cast_gt<to_scalar_at, f64_t>{};

        return result;
    }

    static casts_t make_casts_(scalar_kind_t scalar_kind) {
        switch (scalar_kind) {
        case scalar_kind_t::f64_k: return make_casts_<f64_t>();
        case scalar_kind_t::f32_k: return make_casts_<f32_t>();
        case scalar_kind_t::f16_k: return make_casts_<f16_t>();
        case scalar_kind_t::i8_k: return make_casts_<i8_bits_t>();
        case scalar_kind_t::b1x8_k: return make_casts_<b1x8_t>();
        default: return {};
        }
    }
};

using index_dense_t = index_dense_gt<>;
using index_dense_big_t = index_dense_gt<uuid_t, uint40_t>;

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

    typename men_label_at,   //
    typename women_label_at, //
    typename men_slot_at,    //
    typename women_slot_at,  //

    typename man_to_woman_at = dummy_label_to_label_mapping_t, //
    typename woman_to_man_at = dummy_label_to_label_mapping_t, //
    typename executor_at = dummy_executor_t,                   //
    typename progress_at = dummy_progress_t                    //
    >
static join_result_t join(                                      //
    index_dense_gt<men_label_at, men_slot_at> const& men,       //
    index_dense_gt<women_label_at, women_slot_at> const& women, //

    index_join_config_t config = {},                    //
    man_to_woman_at&& man_to_woman = man_to_woman_at{}, //
    woman_to_man_at&& woman_to_man = woman_to_man_at{}, //
    executor_at&& executor = executor_at{},             //
    progress_at&& progress = progress_at{}) noexcept {

    return men.join(                                 //
        women, config,                               //
        std::forward<woman_to_man_at>(woman_to_man), //
        std::forward<man_to_woman_at>(man_to_woman), //
        std::forward<executor_at>(executor),         //
        std::forward<progress_at>(progress));
}

} // namespace usearch
} // namespace unum
