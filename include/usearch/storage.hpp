#pragma once

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

namespace unum {
namespace usearch {

template <typename key_at, typename compressed_slot_at,        //
          typename tape_allocator_at = std::allocator<byte_t>, //
          typename vectors_allocator_at = tape_allocator_at>   //
class storage_v2 {
    using node_t = node_at<key_at, compressed_slot_at>;
    using nodes_t = std::vector<node_t>;
    using vectors_t = std::vector<byte_t*>;
    using nodes_mutexes_t = bitset_gt<>;

    nodes_t nodes_{};

    /// @brief For every managed `compressed_slot_t` stores a pointer to the allocated vector copy.
    vectors_t vectors_lookup_{};
    /// @brief  Mutex, that limits concurrent access to `nodes_`.
    mutable nodes_mutexes_t nodes_mutexes_{};
    precomputed_constants_t pre_{};
    tape_allocator_at tape_allocator_{};
    /// @brief Allocator for the copied vectors, aligned to widest double-precision scalars.
    vectors_allocator_at vectors_allocator_{};
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
    storage_v2(index_config_t config, tape_allocator_at tape_allocator = {})
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

    using lock_type = node_lock_t;

    bool reserve(std::size_t count) {
        if (count < nodes_.size())
            return true;
        nodes_mutexes_t new_mutexes = nodes_mutexes_t(count);
        nodes_mutexes_ = std::move(new_mutexes);
        nodes_.resize(count);
        vectors_lookup_.resize(count);
        return true;
    }

    /*
    void clear() noexcept {
        if (!has_reset<tape_allocator_t>()) {
            std::size_t n = nodes_count_;
            for (std::size_t i = 0; i != n; ++i)
                node_free_(i);
        } else
            tape_allocator_.deallocate(nullptr, 0);
        nodes_count_ = 0;
        max_level_ = -1;
        entry_slot_ = 0u;
    }
    ****/
    void clear() {
        if (nodes_.data()) {
            std::fill(nodes_.begin(), nodes_.end(), node_t{});
            //   std::fill(vectors_lookup_.begin(), vectors_lookup_.end(), nullptr);
        }
    }
    void reset() {
        nodes_mutexes_ = {};
        nodes_.clear();
        nodes_.shrink_to_fit();

        // vectors_lookup_.clear();
        // vectors_lookup_.shrink_to_fit();
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

    void node_store(size_t slot, node_t node) noexcept { nodes_[slot] = node; }
    inline size_t size() { return nodes_.size(); }
    tape_allocator_at const& node_allocator() const noexcept { return tape_allocator_; }
    // dummy lock just to satisfy the interface
    constexpr inline lock_type node_lock(std::size_t slot) const noexcept {
        while (nodes_mutexes_.atomic_set(slot))
            ;
        return {nodes_mutexes_, slot};
    }
};

} // namespace usearch
} // namespace unum
