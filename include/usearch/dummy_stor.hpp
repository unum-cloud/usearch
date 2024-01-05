
#pragma once

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

namespace unum {
namespace usearch {

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
 **/
template <typename key_at, typename compressed_slot_at,
          typename tape_allocator_at = std::allocator<byte_t>> //
class dummy_storage_single_threaded {
    using node_t = node_at<key_at, compressed_slot_at>;
    using nodes_t = std::vector<node_t>;

    nodes_t nodes_{};
    precomputed_constants_t pre_{};
    tape_allocator_at tape_allocator_{};
    using tape_allocator_traits_t = std::allocator_traits<tape_allocator_at>;
    static_assert(                                                 //
        sizeof(typename tape_allocator_traits_t::value_type) == 1, //
        "Tape allocator must allocate separate addressable bytes");

  public:
    dummy_storage_single_threaded(index_config_t config, tape_allocator_at tape_allocator = {})
        : pre_(node_t::precompute_(config)), tape_allocator_(tape_allocator) {}

    inline node_t get_node_at(std::size_t idx) const noexcept { return nodes_[idx]; }

    inline size_t node_size_bytes(std::size_t idx) const noexcept { return get_node_at(idx).node_size_bytes(pre_); }

    // exported for client-side lock-declaration
    // alternatively, could just use auto in client side
    // ideally, there would be a way to make this "void", but I could not make it work
    // as client side ends up declaring a void variable
    // the downside of passing a primitive like "int" here is the "unused variable" compiler warning
    // for the dummy lock guard variable.
    struct dummy_lock {
        // destructor necessary to avoid "unused variable warning"
        // will this get properly optimized away?
        ~dummy_lock() {}
    };
    using lock_type = dummy_lock;

    bool reserve(std::size_t count) {
        if (count < nodes_.size())
            return true;
        nodes_.resize(count);
        return true;
    }

    void clear() {
        if (nodes_.data())
            std::fill(nodes_.begin(), nodes_.end(), node_t{});
    }
    void reset() {
        nodes_.clear();
        nodes_.shrink_to_fit();
    }

    using span_bytes_t = span_gt<byte_t>;

    span_bytes_t node_malloc(level_t level) noexcept {
        std::size_t node_size = node_t::node_size_bytes(pre_, level);
        byte_t* data = (byte_t*)tape_allocator_.allocate(node_size);
        return data ? span_bytes_t{data, node_size} : span_bytes_t{};
    }
    void node_free(size_t slot, node_t node) {
        if (!has_reset<tape_allocator_at>()) {
            tape_allocator_.deallocate(node.tape(), node.node_size_bytes(pre_));
        } else {
            tape_allocator_.deallocate(nullptr, 0);
        }
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

    void node_store(size_t slot, node_t node) noexcept {
        auto count = nodes_.size();
        nodes_[slot] = node;
    }
    inline size_t size() { return nodes_.size(); }
    tape_allocator_at const& node_allocator() const noexcept { return tape_allocator_; }
    // dummy lock just to satisfy the interface
    constexpr inline lock_type node_lock(std::size_t) const noexcept { return dummy_lock{}; }
};

template <typename key_at, typename compressed_slot_at> class storage_v1 {
    using vector_key_t = key_at;
    using node_t = node_at<vector_key_t, compressed_slot_at>;
    using dynamic_allocator_t = aligned_allocator_gt<byte_t, 64>;
    // using nodes_mutexes_t = bitset_gt<dynamic_allocator_t>;
    using nodes_mutexes_t = bitset_gt<>;
    using nodes_t = std::vector<node_t>;

    index_config_t config_{};
    nodes_t nodes_{};
    /// @brief  Mutex, that limits concurrent access to `nodes_`.
    mutable nodes_mutexes_t nodes_mutexes_{};
};

} // namespace usearch
} // namespace unum
