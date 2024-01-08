
#pragma once

#include <deque>
#include <mutex>
#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>
#include <usearch/storage.hpp>

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
  public:
    using node_t = node_at<key_at, compressed_slot_at>;

  private:
    using nodes_t = std::vector<node_t>;

    nodes_t nodes_{};
    precomputed_constants_t pre_{};
    tape_allocator_at tape_allocator_{};
    memory_mapped_file_t viewed_file_{};
    mutable std::deque<std::mutex> locks_{};
    using tape_allocator_traits_t = std::allocator_traits<tape_allocator_at>;
    static_assert(                                                 //
        sizeof(typename tape_allocator_traits_t::value_type) == 1, //
        "Tape allocator must allocate separate addressable bytes");

  public:
    dummy_storage_single_threaded(index_config_t config, tape_allocator_at tape_allocator = {})
        : pre_(node_t::precompute_(config)), tape_allocator_(tape_allocator) {}

    inline node_t get_node_at(std::size_t idx) const noexcept { return nodes_[idx]; }
    inline byte_t* get_vector_at(std::size_t idx) const noexcept { return nullptr; }
    inline size_t node_size_bytes(std::size_t idx) const noexcept { return get_node_at(idx).node_size_bytes(pre_); }
    bool is_immutable() const noexcept { return bool(viewed_file_); }

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
    using lock_type = std::unique_lock<std::mutex>;

    bool reserve(std::size_t count) {
        if (count < nodes_.size())
            return true;
        nodes_.resize(count);
        locks_.resize(count);
        return true;
    }
    void clear() noexcept {
        if (!is_immutable()) {
            std::size_t n = nodes_.size();
            for (std::size_t i = 0; i != n; ++i) {
                // we do not know which slots have been filled and which ones - no
                // so we iterate over full reserved space
                if (nodes_[i])
                    node_free(i, nodes_[i]);
            }
        }
        if (nodes_.data())
            std::fill(nodes_.begin(), nodes_.end(), node_t{});
    }
    void reset() noexcept { clear(); }

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
    tape_allocator_at const& node_allocator() const noexcept { return tape_allocator_; }
    // dummy lock just to satisfy the interface
    inline lock_type node_lock(std::size_t i) const noexcept { return std::unique_lock(locks_[i]); }

    // serialization

    template <typename output_callback_at, typename vectors_metadata_at>
    serialization_result_t save_vectors_to_stream(output_callback_at& output, std::uint64_t,
                                                  std::uint64_t, //
                                                  const vectors_metadata_at& metadata_buffer,
                                                  serialization_config_t config = {}) const {
        assert(config.exclude_vectors);
        assert(!config.use_64_bit_dimensions);
        bool ok = output(metadata_buffer, sizeof(metadata_buffer));
        assert(ok);
        return {};
    }
    template <typename output_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t save_nodes_to_stream(output_callback_at& output, const index_serialized_header_t& header,
                                                progress_at& = {}) const {
        bool ok = output(&header, sizeof(header));
        assert(ok);
        for (std::size_t i = 0; i != header.size; ++i) {
            node_t node = get_node_at(i);
            level_t level = node.level();
            ok = output(&level, sizeof(level));
            assert(ok);
        }

        // After that dump the nodes themselves
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = get_node_at(i).node_bytes(pre_);
            ok = output(node_bytes.data(), node_bytes.size());
            assert(ok);
        }
        return {};
    }
    template <typename input_callback_at, typename vectors_metadata_at>
    serialization_result_t load_vectors_from_stream(input_callback_at& input, //
                                                    vectors_metadata_at& metadata_buffer,
                                                    serialization_config_t config = {}) {
        assert(config.exclude_vectors);
        assert(!config.use_64_bit_dimensions);
        bool ok = input(metadata_buffer, sizeof(metadata_buffer));
        assert(ok);
        return {};
    }
    template <typename input_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t load_nodes_from_stream(input_callback_at& input, index_serialized_header_t& header,
                                                  progress_at& = {}) noexcept {

        bool ok = input(&header, sizeof(header));
        assert(ok);
        if (!header.size) {
            reset();
            return {};
        }
        buffer_gt<level_t> levels(header.size);
        assert(levels);
        ok = input(levels, header.size * sizeof(level_t));
        assert(ok);

        ok = reserve(header.size);
        assert(ok);

        // Load the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = node_malloc(levels[i]);
            ok = input(node_bytes.data(), node_bytes.size());
            assert(ok);
            node_store(i, node_t{node_bytes.data()});
        }
        return {};
    }
    template <typename vectors_metadata_at>
    serialization_result_t view_vectors_from_stream(
        memory_mapped_file_t& file, //
                                    //// todo!! document that offset is a reference, or better - do not do it this way
        vectors_metadata_at& metadata_buffer, std::size_t& offset, serialization_config_t config = {}) {
        reset();
        assert(config.exclude_vectors);
        assert(!config.use_64_bit_dimensions);

        serialization_result_t result = file.open_if_not();
        assert(result);
        std::memcpy(metadata_buffer, file.data() + offset, sizeof(metadata_buffer));
        offset += sizeof(metadata_buffer);
        return {};
    }
    template <typename progress_at = dummy_progress_t>
    serialization_result_t view_nodes_from_stream(memory_mapped_file_t file, index_serialized_header_t& header,
                                                  std::size_t offset = 0, progress_at& progress = {}) noexcept {
        serialization_result_t result = file.open_if_not();
        std::memcpy(&header, file.data() + offset, sizeof(header));
        if (!header.size) {
            reset();
            return result;
        }
        index_config_t config;
        config.connectivity = header.connectivity;
        config.connectivity_base = header.connectivity_base;
        pre_ = node_t::precompute_(config);
        buffer_gt<std::size_t> offsets(header.size);
        assert(offsets);
        misaligned_ptr_gt<level_t> levels{(byte_t*)file.data() + offset + sizeof(header)};
        offsets[0u] = offset + sizeof(header) + sizeof(level_t) * header.size;
        for (std::size_t i = 1; i < header.size; ++i)
            offsets[i] = offsets[i - 1] + node_t::node_size_bytes(pre_, levels[i - 1]);
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
        return {};
    }
};

using dummy_dummy_storage = dummy_storage_single_threaded<default_key_t, default_slot_t>;
ASSERT_VALID_STORAGE(dummy_dummy_storage);

} // namespace usearch
} // namespace unum
