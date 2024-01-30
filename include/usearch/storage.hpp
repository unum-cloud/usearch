#pragma once

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

namespace unum {
namespace usearch {

/**
 *  @brief  This macro, `HAS_FUNCTION_TEMPLATE`, is a utility to check at
 *  compile-time whether a given type (CHECK_AT) has a member function with a specific name (NAME_AK), signature
 *  (SIGNATURE_AT=return_at(args_at...)), constness (CONST_AK=const|[empty]), and exception specification
 *  (NOEXCEPT_AK=true|false).
 *
 *  It is based on has_reset_gt template:
 *    1. Replace declval<at> with declval<OPTIONAL_CONST at&> to enforce function const-ness
 *     method: https://stackoverflow.com/questions/30407754/how-to-test-if-a-method-is-const
 *    2. Replace .reset with dynamic NAME_AK to support methods with other names
 *    3. Add option to enforce noexcept
 *  method: https://stackoverflow.com/questions/56510130/unit-test-to-check-for-noexcept-property-for-a-c-method
 *
 *  @param[in] CHECK_AT Placeholder type used within the template instantiation to denote the type to be checked.
 *  @param[in] NAME_AK Name of the member function to be checked for. This name is incorporated in the generated
 *  structure's name and used in the check.
 *  @param[in] SIGNATURE_AT Placeholder for the function signature, employed in specializing the template for function
 *  types.
 *  @param[in] CONST_AK Indicates if the member function should be a const function. This forms part of the function
 *  call signature within the check.
 *  @param[in] NOEXCEPT_AK Indicates if the member function should be noexcept. This affects the check, particularly
 *  important for ensuring exception safety in certain contexts.
 *
 *  generates a structure structure named `has_##NAME_AK##_gt` with a static constexpr boolean member `value`. This
 *  member is true if the specified type has a member function that matches the name, signature, constness, and noexcept
 *  status provided in the macro's arguments. Otherwise, it is false.
 *
 *  @example
 *  Suppose you have a class `Foo` with that has an interface requirement of a const noexcept member function `bar` that
 *  returns an `int` and takes a `const double`. To enforce the interface requirement, if this function exists, is
 *  const, and noexcept, you would instantiate the generated template like so:
 *  ```cpp
 * struct Foo {
 *    // CHECK CATCHES: expected double, got double*
 *    // int bar(const double*) const noexcept { return 42; }
 *    // CHECK CATCHES: wrong const-ness
 *    // int bar(const double) noexcept { return 42; }
 *    // CHECK CATCHES: wrong excempt-ness
 *    // int bar(const double) const { return 42; }
 *    // CHECK CATCHES because required int can be cast to double
 *    // double bar(const double) const noexcept { return 42; }
 *    // CHECK CATHCES wrong returned value
 *    // int* bar(const double) const noexcept { return nullptr; }
 *    // CHECK CATHCES wrong signature
 *    // int bar(const double, int) const { return 42; }
 *    //
 *    // SUCCESS! the invariant  we wanted
 *
 *    int bar(const double) const noexcept { return 42; }
 *
 *    //
 *    // Some PROBLEMS
 *    // CHECK **DOES NOT** CATCH. assertion succeeds
 *    // int bar(const double&) const noexcept { return 42; }
 *    // CHECK **DOES NOT** CATCH. assertion succeeds
 *    // int bar(const double&&) const noexcept { return 42; }
 * };
 *
 * HAS_FUNCTION_TEMPLATE(Foo, bar, int(const double), const, true);
 * static_assert(has_bar_gt<Foo, int(double)>::value);
 *  ```
 * If `Foo` indeed has a const noexcept member function `bar` matching this signature, the static assertion succeeds
 * Otherwise, it will cause a compile failure
 */
#define HAS_FUNCTION_TEMPLATE(CHECK_AT, NAME_AK, SIGNATURE_AT, CONST_AK, NOEXCEPT_AK)                                  \
    template <typename, typename at> struct has_##NAME_AK##_gt {                                                       \
        static_assert(std::integral_constant<at, false>::value,                                                        \
                      "Second template parameter needs to be of function type.");                                      \
    };                                                                                                                 \
                                                                                                                       \
    template <typename check_at, typename return_at, typename... args_at>                                              \
    struct has_##NAME_AK##_gt<check_at, return_at(args_at...)> {                                                       \
      private:                                                                                                         \
        template <typename at>                                                                                         \
        static constexpr auto check(at*) ->                                                                            \
            typename std::is_same<decltype(std::declval<CONST_AK at&>().NAME_AK(std::declval<args_at>()...)),          \
                                  return_at>::type;                                                                    \
        template <typename> static constexpr std::false_type check(...);                                               \
                                                                                                                       \
        template <typename at> static constexpr bool f_is_noexcept(at*) {                                              \
            return noexcept(std::declval<CONST_AK at&>().NAME_AK(std::declval<args_at>()...));                         \
        }                                                                                                              \
                                                                                                                       \
        typedef decltype(check<check_at>(0)) type;                                                                     \
                                                                                                                       \
      public: /*                                    if NOEXCEPT_AK then f_is_noexcept(0)    */                         \
        static constexpr bool value = type::value && (!NOEXCEPT_AK || f_is_noexcept<check_at>(0));                     \
    };

/**
 * This is a wrapper around the macro above that allows getting less cryptic error messages
 * in particular, it:
 * 1. Wraps the defined template in a unique namespace to avoid collisions. If this ends up being used elsewhere,
 *    probably it would be worth it to add a __FILE__ prefix to the namespace name as well
 * 2. Regarless of the requrement, it runs signature check without taking into account const-ness and exception
 *    requirement.
 * 3. Only after the initial signature check succeeds, it takes into acount const and noexcept and runs relevant checks,
 * printing descriptive error messages is the constraints are not satisfied
 *
 * The macro takes the same parameters as the one above
 **/
#define ASSERT_HAS_FUNCTION_GM(CHECK_AT, NAME_AK, SIGNATURE_AT, CONST_AK, NOEXCEPT_AK)                                 \
    /************ check function signature without const or noexcept*/                                                 \
    namespace CHECK_AT##__##NAME_AK {                                                                                  \
        HAS_FUNCTION_TEMPLATE(CHECK_AT, NAME_AK, SIGNATURE_AT, , false)                                                \
    }                                                                                                                  \
    static_assert(CHECK_AT##__##NAME_AK::has_##NAME_AK##_gt<CHECK_AT, SIGNATURE_AT>::value,                            \
                  " Function \"" #CHECK_AT "::" #NAME_AK                                                               \
                  "\" does not exist or does not satisfy storage API signature");                                      \
    /************ check function signature with const requirement but without noexcept*/                               \
    namespace CHECK_AT##__##NAME_AK##_const {                                                                          \
        HAS_FUNCTION_TEMPLATE(CHECK_AT, NAME_AK, SIGNATURE_AT, CONST_AK, false)                                        \
    }                                                                                                                  \
    static_assert(CHECK_AT##__##NAME_AK##_const::has_##NAME_AK##_gt<CHECK_AT, SIGNATURE_AT>::value,                    \
                  " Function \"" #CHECK_AT "::" #NAME_AK                                                               \
                  "\" exists but does not satisfy const-requirement of storage API");                                  \
    /************ check function signature with const and noexcept requirements */                                     \
    namespace CHECK_AT##__##NAME_AK##_const_noexcept {                                                                 \
        HAS_FUNCTION_TEMPLATE(CHECK_AT, NAME_AK, SIGNATURE_AT, CONST_AK, NOEXCEPT_AK)                                  \
    }                                                                                                                  \
    static_assert(                                                                                                     \
        !NOEXCEPT_AK || CHECK_AT##__##NAME_AK##_const_noexcept::has_##NAME_AK##_gt<CHECK_AT, SIGNATURE_AT>::value,     \
        " Function \"" #CHECK_AT "::" #NAME_AK "\" exists but does not satisfy noexcept requirement of storage API")

/** Various commonly used shortcusts for the assertion macro above
 * Note: NOCONST in comments indicates intentional lack of const qualifier
 **/
#define ASSERT_HAS_FUNCTION(CHECK_AT, NAME_AK, SIGNATURE_AT)                                                           \
    ASSERT_HAS_FUNCTION_GM(CHECK_AT, NAME_AK, SIGNATURE_AT, /*NOCONST*/, false)
#define ASSERT_HAS_CONST_FUNCTION(CHECK_AT, NAME_AK, SIGNATURE_AT)                                                     \
    ASSERT_HAS_FUNCTION_GM(CHECK_AT, NAME_AK, SIGNATURE_AT, const, false)
#define ASSERT_HAS_NOEXCEPT_FUNCTION(CHECK_AT, NAME_AK, SIGNATURE_AT)                                                  \
    ASSERT_HAS_FUNCTION_GM(CHECK_AT, NAME_AK, SIGNATURE_AT, /*NOCONST*/, true)
#define ASSERT_HAS_CONST_NOEXCEPT_FUNCTION(CHECK_AT, NAME_AK, SIGNATURE_AT)                                            \
    ASSERT_HAS_FUNCTION_GM(CHECK_AT, NAME_AK, SIGNATURE_AT, const, true)

#define HAS_FUNCTION(CHECK_AT, NAME_AK, SIGNATURE_AT) has_##NAME_AK##_gt<CHECK_AT, SIGNATURE_AT>::value

/**
 *  @brief  An example of what a USearch-Storage-compatible output callback should look like.
 *  The callback is called to store arbitrarily serialized usearch index data in the underlying
 *  storage medium managed in the callback implementation
 *
 */
struct dummy_output_callback_t {
    inline bool operator()(const void* /*source memory*/, std::size_t /*size of the source*/) { return true; }
};

/**
 *  @brief  An example of what a USearch-Storage-compatible input callback should look like.
 *  The callback is called to read arbitrarily serialized usearch index data from the underlying
 *  storage medium managed in the callback implementation
 *
 */
struct dummy_input_callback_t {
    inline bool operator()(void* /*destination memory*/, std::size_t /*size of the destination*/) { return true; }
};

/**
 *  @brief  A dummy metadata buffer used in serialization/deserialization API checks below
 *  An actual index implementation might need to keep some app-level constants in here to be serialized on the
 *  stored index binary, but we do not need its structure for type-checking
 *
 */
struct dummy_vectors_metadata_buffer_t {};

struct index_dense_serialization_config_t {
    // We may not want to fetch the vectors from the same file, or allow attaching them afterwards
    bool exclude_vectors = false;
    bool use_64_bit_dimensions = false;
};

using serialization_config_t = index_dense_serialization_config_t;

/**
 * @brief The macro takes in a usearch Storage-provider type, and makes sure the type provides the necessary interface
 * assumed in usearch internals N.B: the validation does notenforce reference argument types properly Validation
 *succeeds even when in the sertions below an interface is required to take a reference type but the actual
 *implementation takes a copy
 **/
#define ASSERT_VALID_STORAGE(CHECK_AT)                                                                                 \
    ASSERT_HAS_CONST_FUNCTION(CHECK_AT, get_node_at, CHECK_AT::node_t(std::size_t idx));                               \
    ASSERT_HAS_CONST_FUNCTION(CHECK_AT, get_vector_at, byte_t*(std::size_t idx));                                      \
    ASSERT_HAS_CONST_FUNCTION(CHECK_AT, node_size_bytes, std::size_t(std::size_t idx));                                \
    ASSERT_HAS_CONST_NOEXCEPT_FUNCTION(CHECK_AT, is_immutable, bool());                                                \
                                                                                                                       \
    /*Container methods */                                                                                             \
    ASSERT_HAS_FUNCTION(CHECK_AT, reserve, bool(std::size_t count));                                                   \
    ASSERT_HAS_NOEXCEPT_FUNCTION(CHECK_AT, clear, void());                                                             \
    ASSERT_HAS_NOEXCEPT_FUNCTION(CHECK_AT, reset, void());                                                             \
    /*Setters*/                                                                                                        \
    ASSERT_HAS_FUNCTION(CHECK_AT, node_malloc, CHECK_AT::span_bytes_t(level_t level));                                 \
    ASSERT_HAS_FUNCTION(CHECK_AT, node_free, void(std::size_t slot, CHECK_AT::node_t node));                           \
    ASSERT_HAS_FUNCTION(CHECK_AT, node_make, CHECK_AT::node_t(CHECK_AT::key_t key, level_t level));                    \
    ASSERT_HAS_FUNCTION(CHECK_AT, node_store, void(std::size_t slot, CHECK_AT::node_t node));                          \
    ASSERT_HAS_FUNCTION(CHECK_AT, set_vector_at,                                                                       \
                        void(std::size_t idx, const byte_t* vector_data, std::size_t vector_bytes, bool copy_vector,   \
                             bool reuse_node));                                                                        \
    /*Locking*/                                                                                                        \
    ASSERT_HAS_CONST_NOEXCEPT_FUNCTION(CHECK_AT, node_lock, CHECK_AT::lock_type(std::size_t idx));                     \
    /*Save/Restore API enforcement*/                                                                                   \
    ASSERT_HAS_FUNCTION(CHECK_AT, save_vectors_to_stream,                                                              \
                        serialization_result_t(                                                                        \
                            dummy_output_callback_t& cb, std::size_t vector_size_bytes, std::uint64_t node_count,      \
                            const dummy_vectors_metadata_buffer_t& metadata_buffer, serialization_config_t config));   \
    ASSERT_HAS_CONST_FUNCTION(CHECK_AT, save_nodes_to_stream,                                                          \
                              serialization_result_t(dummy_output_callback_t& cb,                                      \
                                                     const index_serialized_header_t& header,                          \
                                                     dummy_progress_t& progress));                                     \
    ASSERT_HAS_FUNCTION(CHECK_AT, load_vectors_from_stream,                                                            \
                        serialization_result_t(dummy_input_callback_t& cb,                                             \
                                               const dummy_vectors_metadata_buffer_t& metadata_buffer,                 \
                                               serialization_config_t config));                                        \
    ASSERT_HAS_FUNCTION(CHECK_AT, load_nodes_from_stream,                                                              \
                        serialization_result_t(dummy_input_callback_t& cb, index_serialized_header_t& header,          \
                                               dummy_progress_t& progress));                                           \
                                                                                                                       \
    /* View from file API*/                                                                                            \
    ASSERT_HAS_FUNCTION(CHECK_AT, view_vectors_from_file,                                                              \
                        serialization_result_t(memory_mapped_file_t& file,                                             \
                                               dummy_vectors_metadata_buffer_t& metadata_buffer, std::size_t& offset,  \
                                               serialization_config_t config));                                        \
    ASSERT_HAS_FUNCTION(CHECK_AT, view_nodes_from_file,                                                                \
                        serialization_result_t(memory_mapped_file_t file, index_serialized_header_t& metadata_buffer,  \
                                               std::size_t& offset, dummy_progress_t& progress));                      \
    static_assert(true, "this is to require a semicolon at the end of macro call")

/** I initially used this abstract class as a way to enforce storage API but ran into several limitations mentioned
 * below I switched to macro+template based approach in the end, but left this around, in case there are ways to work
 * around the issues below that I am not aware of.
 **/
template <typename key_at, typename compressed_slot_at, //
          typename tape_allocator_at,                   //
          typename vectors_allocator_at,                //
          typename dynamic_allocator_at>                //
class storage_interface {
  public:
    using key_t = key_at;
    using node_t = node_at<key_t, compressed_slot_at>;
    // storage_interface(index_config_t conig, tape_allocator_at allocator = {});

    struct lock_type;

    // q:: ask-Ashot can I enforce this interface function in inherited storages somehow?
    // currently impossible because
    // 1. can do virtual constexpr after c++2a
    // 2. making this virtual would enforce this particular lock_type struct as the return type,
    //  and not an equivalently named one in the child class
    // I currently enforice it via macros
    constexpr inline lock_type node_lock(std::size_t slot) const noexcept;

    virtual inline node_t get_node_at(std::size_t idx) const noexcept = 0;
    virtual inline std::size_t node_size_bytes(std::size_t idx) const noexcept = 0;
    virtual inline byte_t* get_vector_at(std::size_t idx) const noexcept = 0;

    inline void set_at(std::size_t idx, node_t node, byte_t* vector_data, std::size_t vector_size, bool reuse_node);

    // the following functions take template arguments so cannot be type-enforced via virtual function inheritence
    // as far as I can tell.
    // virtual void load_vectors_from_stream() = 0;
    // virtual void load_nodes_from_stream() = 0;

    // serialization_result_t save_vectors_to_stream() const;
    // serialization_result_t save_nodes_to_stream() const;

    // serialization_result_t view_vectors_from_file() const;
    // serialization_result_t view_nodes_from_file() const;

    virtual std::size_t size() const noexcept = 0;
    virtual bool reserve(std::size_t count) = 0;
    virtual void clear() noexcept = 0;
    virtual void reset() noexcept = 0;
    std::size_t memory_usage();
};

/*Default allocators for storage_v2 */
using dynamic_allocator_t = aligned_allocator_gt<byte_t, 64>;
using tape_allocator_t = memory_mapping_allocator_gt<64>;
using vectors_tape_allocator_t = memory_mapping_allocator_gt<8>;
/**
 * @brief   Storage abstraction for HNSW graph and associated vector data
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
 *  @tparam tape_allocator_at
 *      Potentially different memory allocator for primary allocations of nodes and vectors.
 *      It would never `deallocate` separate entries, and would only free all the space at once.
 *      The allocated buffers may be uninitialized.
 *
 * NOTE:
 * The class below used to inherit from storage_interface via:
 * class storage_v2 : public storage_interface<key_at, compressed_slot_at, tape_allocator_at, vectors_allocator_at,
 *                                            dynamic_allocator_at>
 * I disabled inheritence for now as interface compatibility is more
 * thoroughly enforced via the macros at the beginning of this file
 **/
template <typename key_at, typename compressed_slot_at,             //
          typename tape_allocator_at = tape_allocator_t,            //
          typename vectors_allocator_at = vectors_tape_allocator_t, //
          typename dynamic_allocator_at = dynamic_allocator_t>      //
class storage_v2_at {
  public:
    using key_t = key_at;
    using node_t = node_at<key_t, compressed_slot_at>;

  private:
    using nodes_mutexes_t = bitset_gt<dynamic_allocator_at>;
    using dynamic_allocator_traits_t = std::allocator_traits<dynamic_allocator_at>;
    using levels_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<level_t>;
    using nodes_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<node_t>;
    using offsets_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<std::size_t>;
    using vectors_allocator_t = typename dynamic_allocator_traits_t::template rebind_alloc<byte_t*>;
    using nodes_t = buffer_gt<node_t, nodes_allocator_t>;
    // todo:: ask-Ashot: in the older version vectors_lookup_ was using the default vector allocator,
    // and not the dynamic_allocator_at that was passed it.
    // Can remove this if the previous approach was intentional
    // Update (Jan 10): It seems giving vectors_allocator_t as vectors_t
    // allocator below only works when CMAKE_HAVE_LIBC_PTHREAD is false
    // otherwise, I get a compile error
    using vectors_t = std::vector<byte_t*>;

    /// @brief  C-style array of `node_t` smart-pointers.
    // buffer_gt<node_t, nodes_allocator_t> nodes_{};

    nodes_t nodes_{};

    /// @brief For every managed `compressed_slot_t` stores a pointer to the allocated vector copy.
    vectors_t vectors_lookup_{};
    /// @brief  Mutex, that limits concurrent access to `nodes_`.
    mutable nodes_mutexes_t nodes_mutexes_{};
    precomputed_constants_t pre_{};
    tape_allocator_at tape_allocator_{};
    /// @brief Allocator for the copied vectors, aligned to widest double-precision scalars.
    vectors_allocator_at vectors_allocator_{};

    std::uint64_t matrix_rows_ = 0;
    std::uint64_t matrix_cols_ = 0;
    bool vectors_loaded_{};
    memory_mapped_file_t viewed_file_{};
    using tape_allocator_traits_t = std::allocator_traits<tape_allocator_at>;
    static_assert(                                                 //
        sizeof(typename tape_allocator_traits_t::value_type) == 1, //
        "Tape allocator must allocate separate addressable bytes");

    struct node_lock_t {
        nodes_mutexes_t& mutexes;
        std::size_t slot;
        inline ~node_lock_t() noexcept { mutexes.atomic_reset(slot); }
    };

  public:
    storage_v2_at(index_config_t config, tape_allocator_at tape_allocator = {})
        : pre_(node_t::precompute_(config)), tape_allocator_(tape_allocator) {}

    inline node_t get_node_at(std::size_t idx) const noexcept { return nodes_[idx]; }
    // todo:: most of the time this is called for const* vector, maybe add a separate interface for const?
    inline byte_t* get_vector_at(std::size_t idx) const noexcept { return vectors_lookup_[idx]; }
    inline void set_vector_at(std::size_t idx, const byte_t* vector_data, std::size_t bytes_per_vector,
                              bool copy_vector, bool reuse_node) {
        usearch_assert_m(!(reuse_node && !copy_vector),
                         "Cannot reuse node when not copying as there is no allocation needed");
        if (copy_vector) {
            if (!reuse_node)
                vectors_lookup_[idx] = vectors_allocator_.allocate(bytes_per_vector);
            std::memcpy(vectors_lookup_[idx], vector_data, bytes_per_vector);
        } else
            vectors_lookup_[idx] = (byte_t*)vector_data;
    }

    inline size_t node_size_bytes(std::size_t idx) const noexcept { return get_node_at(idx).node_size_bytes(pre_); }
    bool is_immutable() const noexcept { return bool(viewed_file_); }

    using lock_type = node_lock_t;

    bool reserve(std::size_t count) {
        if (count < nodes_.size() && count < nodes_mutexes_.size())
            return true;
        nodes_mutexes_t new_mutexes(count);
        nodes_t new_nodes(count);
        if (!new_mutexes || !new_nodes)
            return false;
        if (nodes_)
            std::memcpy(new_nodes.data(), nodes_.data(), sizeof(node_t) * nodes_.size());

        nodes_mutexes_ = std::move(new_mutexes);
        nodes_ = std::move(new_nodes);
        // todo:: make sure to only reserve this if vectors are not stored externally
        // will probably need to pass the fact as storage config parameter
        vectors_lookup_.resize(count);
        return true;
    }

    void clear() noexcept {
        if (!viewed_file_) {
            if (!has_reset<tape_allocator_at>()) {
                std::size_t n = nodes_.size();
                for (std::size_t i = 0; i != n; ++i) {
                    // we do not know which slots have been filled and which ones - no
                    // so we iterate over full reserved space
                    if (nodes_[i])
                        node_free(i, nodes_[i]);
                }
            } else
                tape_allocator_.deallocate(nullptr, 0);

            if (!has_reset<vectors_allocator_at>()) {
                std::size_t n = vectors_lookup_.size();
                for (std::size_t i = 0; i != n; ++i) {
                    if (vectors_lookup_[i])
                        vectors_allocator_.deallocate(vectors_lookup_[i], matrix_cols_);
                }
            } else
                tape_allocator_.deallocate(nullptr, 0);
        }
        std::fill(nodes_.begin(), nodes_.end(), node_t{});
        viewed_file_ = {};
    }

    void reset() noexcept {
        nodes_mutexes_ = {};
        nodes_ = {};

        vectors_lookup_.clear();
        vectors_lookup_.shrink_to_fit();
        viewed_file_ = {};
    }

    using span_bytes_t = span_gt<byte_t>;

    span_bytes_t node_malloc(level_t level) noexcept {
        std::size_t node_size = node_t::node_size_bytes(pre_, level);
        byte_t* data = (byte_t*)tape_allocator_.allocate(node_size);
        return data ? span_bytes_t{data, node_size} : span_bytes_t{};
    }
    void node_free(size_t slot, node_t node) {
        tape_allocator_.deallocate(node.tape(), node.node_size_bytes(pre_));
        nodes_[slot] = node_t{};
    }
    node_t node_make(key_at key, level_t level) noexcept {
        span_bytes_t node_bytes = node_malloc(level);
        if (!node_bytes)
            return {};

        std::memset(node_bytes.data(), 0, node_bytes.size());
        node_t node{(byte_t*)node_bytes.data()};
        node.key(key);
        node.level(level);
        return node;
    }

    // node_t node_make_copy_(span_bytes_t old_bytes) noexcept {
    //     byte_t* data = (byte_t*)tape_allocator_.allocate(old_bytes.size());
    //     if (!data)
    //         return {};
    //     std::memcpy(data, old_bytes.data(), old_bytes.size());
    //     return node_t{data};
    // }

    void node_store(size_t slot, node_t node) noexcept { nodes_[slot] = node; }
    tape_allocator_at const& node_allocator() const noexcept { return tape_allocator_; }
    // dummy lock just to satisfy the interface
    constexpr inline lock_type node_lock(std::size_t slot) const noexcept {
        while (nodes_mutexes_.atomic_set(slot))
            ;
        return {nodes_mutexes_, slot};
    }

#pragma region Storage Serialization and Deserialization

    /**
     *  @brief  Saves serialized binary index vectors to a stream.
     *  @param[in] output Output stream to which vectors will be saved to according to this storage format.
     *  @param[in] metadata_buffer A buffer opaque to Storage, that will be serialized into output stream
     *  @param[in] config Configuration parameters for imports.
     *  @return Outcome descriptor explicitly convertible to boolean.
     */
    template <typename output_callback_at, typename vectors_metadata_at>
    serialization_result_t save_vectors_to_stream(output_callback_at& output, std::uint64_t vector_size_bytes,
                                                  std::uint64_t node_count, //
                                                  const vectors_metadata_at& metadata_buffer,
                                                  serialization_config_t config = {}) const {

        serialization_result_t result;
        std::uint64_t matrix_rows = 0;
        std::uint64_t matrix_cols = 0;

        // We may not want to put the vectors into the same file
        if (!config.exclude_vectors) {
            // Save the matrix size
            if (!config.use_64_bit_dimensions) {
                std::uint32_t dimensions[2];
                dimensions[0] = static_cast<std::uint32_t>(node_count);
                dimensions[1] = static_cast<std::uint32_t>(vector_size_bytes);
                if (!output(&dimensions, sizeof(dimensions)))
                    return result.failed("Failed to serialize into stream");
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            } else {
                std::uint64_t dimensions[2];
                dimensions[0] = static_cast<std::uint64_t>(node_count);
                dimensions[1] = static_cast<std::uint64_t>(vector_size_bytes);
                if (!output(&dimensions, sizeof(dimensions)))
                    return result.failed("Failed to serialize into stream");
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            }

            // Dump the vectors one after another
            for (std::uint64_t i = 0; i != matrix_rows; ++i) {
                const byte_t* vector = get_vector_at(i);
                if (!output(vector, matrix_cols))
                    return result.failed("Failed to serialize into stream");
            }
        }

        if (!output(&metadata_buffer, sizeof(metadata_buffer)))
            return result.failed("Failed to read the index vector metadata");

        return result;
    }

    /**
     *  @brief  Symmetric to `save_from_stream`, pulls data from a stream.
     */
    template <typename output_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t save_nodes_to_stream(output_callback_at& output, const index_serialized_header_t& header,
                                                progress_at& progress = {}) const {

        serialization_result_t result;

        if (!output(&header, sizeof(header)))
            return result.failed("Failed to serialize the header into stream");

        // Progress status
        std::size_t processed = 0;
        std::size_t const total = 2 * header.size;

        // Export the number of levels per node
        // That is both enough to estimate the overall memory consumption,
        // and to be able to estimate the offsets of every entry in the file.
        for (std::size_t i = 0; i != header.size; ++i) {
            node_t node = get_node_at(i);
            level_t level = node.level();
            if (!output(&level, sizeof(level)))
                return result.failed("Failed to serialize into stream");
            if (!progress(++processed, total))
                return result.failed("Terminated by user");
        }

        // After that dump the nodes themselves
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = get_node_at(i).node_bytes(pre_);
            if (!output(node_bytes.data(), node_bytes.size()))
                return result.failed("Failed to serialize into stream");
            if (!progress(++processed, total))
                return result.failed("Terminated by user");
        }
        return result;
    }

    /**
     *  @brief Parses the index from file to RAM.
     *  @param[in] input Input stream from which vectors will be loaded according to this storage format.
     *  @param[out] metadata_buffer A buffer opaque to Storage, into which previously stored metadata will be
     *  loaded from input stream
     *  @param[in] config Configuration parameters for imports.
     *  @return Outcome descriptor explicitly convertible to boolean.
     */
    template <typename input_callback_at, typename vectors_metadata_at>
    serialization_result_t load_vectors_from_stream(input_callback_at& input, //
                                                    vectors_metadata_at& metadata_buffer,
                                                    serialization_config_t config = {}) {

        reset();

        // Infer the new index size
        serialization_result_t result;
        std::uint64_t matrix_rows = 0;
        std::uint64_t matrix_cols = 0;

        // We may not want to load the vectors from the same file, or allow attaching them afterwards
        if (!config.exclude_vectors) {
            // Save the matrix size
            if (!config.use_64_bit_dimensions) {
                std::uint32_t dimensions[2];
                if (!input(&dimensions, sizeof(dimensions)))
                    return result.failed("Failed to read 32-bit dimensions of the matrix");
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            } else {
                std::uint64_t dimensions[2];
                if (!input(&dimensions, sizeof(dimensions)))
                    return result.failed("Failed to read 64-bit dimensions of the matrix");
                matrix_rows = dimensions[0];
                matrix_cols = dimensions[1];
            }
            // Load the vectors one after another
            // most of this logic should move within storage class
            reserve(matrix_rows);
            for (std::uint64_t slot = 0; slot != matrix_rows; ++slot) {
                byte_t* vector = vectors_allocator_.allocate(matrix_cols);
                if (!input(vector, matrix_cols))
                    return result.failed("Failed to read vectors");
                vectors_lookup_[slot] = vector;
            }
            vectors_loaded_ = true;
        }
        matrix_rows_ = matrix_rows;
        matrix_cols_ = matrix_cols;

        if (!input(metadata_buffer, sizeof(metadata_buffer)))
            return result.failed("Failed to read the index vector metadata");

        return result;
    }

    /**
     *  @brief  Symmetric to `save_from_stream`, pulls data from a stream.
     */
    template <typename input_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t load_nodes_from_stream(input_callback_at& input, index_serialized_header_t& header,
                                                  progress_at& progress = {}) noexcept {

        serialization_result_t result;

        // Pull basic metadata directly into the return paramter
        if (!input(&header, sizeof(header)))
            return result.failed("Failed to pull the header from the stream");

        // We are loading an empty index, no more work to do
        if (!header.size) {
            reset();
            return result;
        }

        // Allocate some dynamic memory to read all the levels
        buffer_gt<level_t, levels_allocator_t> levels(header.size);
        if (!levels)
            return result.failed("Out of memory");
        if (!input(levels, header.size * sizeof(level_t)))
            return result.failed("Failed to pull nodes levels from the stream");

        if (!reserve(header.size)) {
            reset();
            return result.failed("Out of memory");
        }

        // Load the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = node_malloc(levels[i]);
            if (!input(node_bytes.data(), node_bytes.size())) {
                reset();
                return result.failed("Failed to pull nodes from the stream");
            }
            node_store(i, node_t{node_bytes.data()});

            if (!progress(i + 1, header.size))
                return result.failed("Terminated by user");
        }

        if (vectors_loaded_ && header.size != static_cast<std::size_t>(matrix_rows_))
            return result.failed("Index size and the number of vectors doesn't match");
        return {};
    }

    /**
     *  @brief Parses the index from file to RAM.
     *  @param[in] file Memory mapped file from which vectors will be viewed according to this storage format.
     *  @param[out] metadata_buffer A buffer opaque to Storage, into which previously stored metadata will be
     *  loaded from input stream
     *  @param[in] config Configuration parameters for imports.
     *  @return Outcome descriptor explicitly convertible to boolean.
     */
    template <typename vectors_metadata_at>
    serialization_result_t view_vectors_from_file(
        memory_mapped_file_t& file, //
                                    //// todo!! document that offset is a reference, or better - do not do it this way
        vectors_metadata_at& metadata_buffer, std::size_t& offset, serialization_config_t config = {}) {

        reset();

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        // Infer the new index size
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
            vectors_loaded_ = true;
        }
        matrix_rows_ = matrix_rows;
        matrix_cols_ = matrix_cols;
        // q:: how does this work when vectors are excluded?
        // Address the vectors
        reserve(matrix_rows);
        if (!config.exclude_vectors)
            for (std::uint64_t slot = 0; slot != matrix_rows; ++slot)
                set_vector_at(slot, vectors_buffer.data() + matrix_cols * slot, matrix_cols, //
                              false, false);

        if (file.size() - offset < sizeof(metadata_buffer))
            return result.failed("File is corrupted and lacks a header");

        std::memcpy(metadata_buffer, file.data() + offset, sizeof(metadata_buffer));
        offset += sizeof(metadata_buffer);

        return result;
    }

    /**
     *  @brief  Symmetric to `save_from_stream`, pulls data from a stream.
     */
    template <typename progress_at = dummy_progress_t>
    serialization_result_t view_nodes_from_file(memory_mapped_file_t file, index_serialized_header_t& header,
                                                std::size_t offset = 0, progress_at& progress = {}) noexcept {

        serialization_result_t result = file.open_if_not();
        if (!result)
            return result;

        // Pull basic metadata
        if (file.size() - offset < sizeof(header))
            return result.failed("File is corrupted and lacks a header");
        std::memcpy(&header, file.data() + offset, sizeof(header));

        if (!header.size) {
            reset();
            return result;
        }

        // update config_ and pre_ for correct node_t size calculations below
        index_config_t config;
        config.connectivity = header.connectivity;
        config.connectivity_base = header.connectivity_base;
        pre_ = node_t::precompute_(config);

        buffer_gt<std::size_t, offsets_allocator_t> offsets(header.size);

        if (!offsets)
            return result.failed("Out of memory");

        // before mapping levels[] from file, let's make sure the file is large enough
        if (file.size() - offset - sizeof(header) - header.size * sizeof(level_t) < 0)
            return result.failed("File is corrupted. Unable to parse node levels from file");

        misaligned_ptr_gt<level_t> levels{(byte_t*)file.data() + offset + sizeof(header)};
        offsets[0u] = offset + sizeof(header) + sizeof(level_t) * header.size;

        for (std::size_t i = 1; i < header.size; ++i)
            offsets[i] = offsets[i - 1] + node_t::node_size_bytes(pre_, levels[i - 1]);

        std::size_t total_bytes = offsets[header.size - 1] + node_t::node_size_bytes(pre_, levels[header.size - 1]);
        if (file.size() < total_bytes) {
            reset();
            return result.failed("File is corrupted and can't fit all the nodes");
        }

        if (!reserve(header.size)) {
            reset();
            return result.failed("Out of memory");
        }

        // Rapidly address all the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            node_store(i, node_t{(byte_t*)file.data() + offsets[i]});
            if (!progress(i + 1, header.size))
                return result.failed("Terminated by user");
        }
        viewed_file_ = std::move(file);

        if (vectors_loaded_ && header.size != static_cast<std::size_t>(matrix_rows_))
            return result.failed("Index size and the number of vectors doesn't match");

        return {};
    }

#pragma endregion
};

using default_storage_v2_t = storage_v2_at<default_key_t, default_slot_t>;

ASSERT_VALID_STORAGE(default_storage_v2_t);

} // namespace usearch
} // namespace unum
