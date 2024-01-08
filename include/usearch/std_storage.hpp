
#pragma once

#include <deque>
#include <mutex>
#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>
#include <usearch/storage.hpp>

namespace unum {
namespace usearch {

/**
 * @brief  A simple Storage implementation that uses standard cpp containers and complies with the usearch storage
 *abstraction for HNSW graph and associated vector data
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
template <typename key_at, typename compressed_slot_at, typename allocator_at = std::allocator<byte_t>> //
class std_storage_at {
  public:
    using node_t = node_at<key_at, compressed_slot_at>;

  private:
    using nodes_t = std::vector<node_t>;
    using span_bytes_t = span_gt<byte_t>;
    using vectors_t = std::vector<span_bytes_t>;

    nodes_t nodes_{};
    vectors_t vectors_{};
    precomputed_constants_t pre_{};
    allocator_at allocator_{};
    static_assert(!has_reset<allocator_at>(), "reset()-able memory allocators not supported for this storage provider");
    memory_mapped_file_t viewed_file_{};
    mutable std::deque<std::mutex> locks_{};
    // the next three are used only in serialization/deserialization routines to know how to serialize vectors
    // since this is only for serde/vars are marked mutable to still allow const-ness of saving method interface on
    // storage instance
    mutable size_t node_count_{};
    mutable size_t vector_size_{};
    // defaulted to true because that is what test.cpp assumes when using this storage directly
    mutable bool exclude_vectors_ = true;

    // used in place of error handling throughout the class
    static void expect(bool must_be_true) {
        if (!must_be_true)
            throw std::runtime_error("Failed!");
    }

  public:
    std_storage_at(index_config_t config, allocator_at allocator = {})
        : pre_(node_t::precompute_(config)), allocator_(allocator) {}

    inline node_t get_node_at(std::size_t idx) const noexcept { return nodes_[idx]; }
    inline byte_t* get_vector_at(std::size_t idx) const noexcept { return vectors_[idx].data(); }
    inline size_t node_size_bytes(std::size_t idx) const noexcept { return get_node_at(idx).node_size_bytes(pre_); }
    bool is_immutable() const noexcept { return bool(viewed_file_); }

    /* To get a single-threaded implementation of storage with no locking, replace lock_type
     *  with the following and return dummy_lock{} from node_lock()
     *      struct dummy_lock {
     *          // destructor necessary to avoid "unused variable warning"
     *          // at callcites of node_lock()
     *          ~dummy_lock() = default;
     *      };
     *      using lock_type = dummy_lock;
     */
    using lock_type = std::unique_lock<std::mutex>;

    bool reserve(std::size_t count) {
        if (count < nodes_.size())
            return true;
        nodes_.resize(count);
        vectors_.resize(count);
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
            n = vectors_.size();
            for (std::size_t i = 0; i != n; ++i) {
                span_bytes_t v = vectors_[i];
                if (v.data()) {
                    allocator_.deallocate(v.data(), v.size());
                }
            }
        }
        if (vectors_.data())
            std::fill(vectors_.begin(), vectors_.end(), span_bytes_t{});
        if (nodes_.data())
            std::fill(nodes_.begin(), nodes_.end(), node_t{});
    }
    void reset() noexcept { clear(); }

    span_bytes_t node_malloc(level_t level) noexcept {
        std::size_t node_size = node_t::node_size_bytes(pre_, level);
        byte_t* data = (byte_t*)allocator_.allocate(node_size);
        return data ? span_bytes_t{data, node_size} : span_bytes_t{};
    }
    void node_free(size_t slot, node_t node) {
        allocator_.deallocate(node.tape(), node.node_size_bytes(pre_));
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
    void node_store(size_t slot, node_t node) noexcept { nodes_[slot] = node; }
    void set_vector_at(size_t slot, const byte_t* vector_data, size_t vector_size, bool copy_vector, bool reuse_node) {

        usearch_assert_m(!(reuse_node && !copy_vector),
                         "Cannot reuse node when not copying as there is no allocation needed");
        if (copy_vector) {
            if (!reuse_node)
                vectors_[slot] = span_bytes_t{allocator_.allocate(vector_size), vector_size};
            std::memcpy(vectors_[slot].data(), vector_data, vector_size);
        } else
            vectors_[slot] = span_bytes_t{(byte_t*)vector_data, vector_size};
    }

    allocator_at const& node_allocator() const noexcept { return allocator_; }

    inline lock_type node_lock(std::size_t i) const noexcept { return std::unique_lock(locks_[i]); }

    // serialization

    template <typename output_callback_at, typename vectors_metadata_at>
    serialization_result_t save_vectors_to_stream(output_callback_at& output, std::uint64_t vector_size_bytes,
                                                  std::uint64_t node_count, //
                                                  const vectors_metadata_at& metadata_buffer,
                                                  serialization_config_t config = {}) const {
        expect(!config.use_64_bit_dimensions);
        expect(output(metadata_buffer, sizeof(metadata_buffer)));

        vector_size_ = vector_size_bytes;
        node_count_ = node_count;
        exclude_vectors_ = config.exclude_vectors;
        return {};
    }

    template <typename output_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t save_nodes_to_stream(output_callback_at& output, const index_serialized_header_t& header,
                                                progress_at& = {}) const {
        expect(output(&header, sizeof(header)));
        expect(output(&vector_size_, sizeof(vector_size_)));
        expect(output(&node_count_, sizeof(node_count_)));
        for (std::size_t i = 0; i != header.size; ++i) {
            node_t node = get_node_at(i);
            level_t level = node.level();
            expect(output(&level, sizeof(level)));
        }

        // After that dump the nodes themselves
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = get_node_at(i).node_bytes(pre_);
            expect(output(node_bytes.data(), node_bytes.size()));
            if (!exclude_vectors_) {
                byte_t* vector_bytes = get_vector_at(i);
                expect(output(vector_bytes, vector_size_));
            }
        }
        return {};
    }

    template <typename input_callback_at, typename vectors_metadata_at>
    serialization_result_t load_vectors_from_stream(input_callback_at& input, //
                                                    vectors_metadata_at& metadata_buffer,
                                                    serialization_config_t config = {}) {
        expect(!config.use_64_bit_dimensions);
        expect(input(metadata_buffer, sizeof(metadata_buffer)));
        exclude_vectors_ = config.exclude_vectors;
        return {};
    }

    template <typename input_callback_at, typename progress_at = dummy_progress_t>
    serialization_result_t load_nodes_from_stream(input_callback_at& input, index_serialized_header_t& header,
                                                  progress_at& = {}) noexcept {
        expect(input(&header, sizeof(header)));
        expect(input(&vector_size_, sizeof(vector_size_)));
        expect(input(&node_count_, sizeof(node_count_)));
        if (!header.size) {
            reset();
            return {};
        }
        buffer_gt<level_t> levels(header.size);
        expect(levels);
        expect(input(levels, header.size * sizeof(level_t)));
        expect(reserve(header.size));

        // Load the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            span_bytes_t node_bytes = node_malloc(levels[i]);
            expect(input(node_bytes.data(), node_bytes.size()));
            node_store(i, node_t{node_bytes.data()});
            if (!exclude_vectors_) {
                byte_t* vector_bytes = allocator_.allocate(vector_size_);
                expect(input(vector_bytes, vector_size_));
                set_vector_at(i, vector_bytes, vector_size_, false, false);
            }
        }
        return {};
    }

    template <typename vectors_metadata_at>
    serialization_result_t view_vectors_from_stream(
        memory_mapped_file_t& file, //
                                    //// todo!! document that offset is a reference, or better - do not do it this way
        vectors_metadata_at& metadata_buffer, std::size_t& offset, serialization_config_t config = {}) {
        reset();
        exclude_vectors_ = config.exclude_vectors;
        expect(!config.use_64_bit_dimensions);

        expect(bool(file.open_if_not()));
        std::memcpy(metadata_buffer, file.data() + offset, sizeof(metadata_buffer));
        offset += sizeof(metadata_buffer);
        return {};
    }

    template <typename progress_at = dummy_progress_t>
    serialization_result_t view_nodes_from_stream(memory_mapped_file_t file, index_serialized_header_t& header,
                                                  std::size_t offset = 0, progress_at& = {}) noexcept {
        serialization_result_t result = file.open_if_not();
        std::memcpy(&header, file.data() + offset, sizeof(header));
        offset += sizeof(header);
        std::memcpy(&vector_size_, file.data() + offset, sizeof(vector_size_));
        offset += sizeof(vector_size_);
        std::memcpy(&node_count_, file.data() + offset, sizeof(node_count_));
        offset += sizeof(node_count_);
        if (!header.size) {
            reset();
            return result;
        }
        index_config_t config;
        config.connectivity = header.connectivity;
        config.connectivity_base = header.connectivity_base;
        pre_ = node_t::precompute_(config);
        buffer_gt<std::size_t> offsets(header.size);
        expect(offsets);
        misaligned_ptr_gt<level_t> levels{(byte_t*)file.data() + offset};
        offset += sizeof(level_t) * header.size;
        offsets[0u] = offset;
        for (std::size_t i = 1; i < header.size; ++i)
            offsets[i] = offsets[i - 1] + node_t::node_size_bytes(pre_, levels[i - 1]) + vector_size_;
        expect(reserve(header.size));

        // Rapidly address all the nodes
        for (std::size_t i = 0; i != header.size; ++i) {
            node_store(i, node_t{(byte_t*)file.data() + offsets[i]});
            set_vector_at(i, (byte_t*)file.data() + offsets[i] + node_size_bytes(i), vector_size_, false, false);
        }
        viewed_file_ = std::move(file);
        return {};
    }
};

using default_std_storage_t = std_storage_at<default_key_t, default_slot_t>;

template <typename key_at, typename slot_at> using default_allocator_std_storage_at = std_storage_at<key_at, slot_at>;
ASSERT_VALID_STORAGE(default_std_storage_t);

} // namespace usearch
} // namespace unum
