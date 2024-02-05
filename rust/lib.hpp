#pragma once
#include "rust/cxx.h"

#include <memory> // `std::shared_ptr`

#include <usearch/index_dense.hpp>
#include <usearch/lantern_storage.hpp>

struct Matches;
struct IndexOptions;
using namespace unum::usearch;
class NativeIndex {

  public:
    using metric_t = unum::usearch::metric_punned_t;
    using distance_t = unum::usearch::distance_punned_t;
    using index_dense_t = index_dense_gt<default_key_t, default_slot_t, lantern_internal_storage_t>;
    using add_result_t = typename index_dense_t::add_result_t;
    using search_result_t = typename index_dense_t::search_result_t;
    using vector_key_t = typename index_dense_t::vector_key_t;

    NativeIndex(std::unique_ptr<index_dense_t> index);

    void reserve(size_t) const;

    void add_i8(vector_key_t key, rust::Slice<int8_t const> vector) const;
    void add_f16(vector_key_t key, rust::Slice<uint16_t const> vector) const;
    void add_f32(vector_key_t key, rust::Slice<float const> vector) const;
    void add_f64(vector_key_t key, rust::Slice<double const> vector) const;

    Matches search_i8(rust::Slice<int8_t const> vector, size_t count) const;
    Matches search_f16(rust::Slice<uint16_t const> vector, size_t count) const;
    Matches search_f32(rust::Slice<float const> vector, size_t count) const;
    Matches search_f64(rust::Slice<double const> vector, size_t count) const;

    size_t get_i8(vector_key_t key, rust::Slice<int8_t> vector) const;
    size_t get_f16(vector_key_t key, rust::Slice<uint16_t> vector) const;
    size_t get_f32(vector_key_t key, rust::Slice<float> vector) const;
    size_t get_f64(vector_key_t key, rust::Slice<double> vector) const;

    size_t expansion_add() const;
    size_t expansion_search() const;
    void change_expansion_add(size_t n) const;
    void change_expansion_search(size_t n) const;

    size_t count(vector_key_t key) const;
    size_t remove(vector_key_t key) const;
    size_t rename(vector_key_t from, vector_key_t to) const;
    bool contains(vector_key_t key) const;

    size_t dimensions() const;
    size_t connectivity() const;
    size_t size() const;
    size_t capacity() const;
    size_t serialized_length() const;

    void save(rust::Str path) const;
    void load(rust::Str path) const;
    void view(rust::Str path) const;
    void reset() const;
    size_t memory_usage() const;

    void save_to_buffer(rust::Slice<uint8_t> buffer) const;
    void load_from_buffer(rust::Slice<uint8_t const> buffer) const;
    void view_from_buffer(rust::Slice<uint8_t const> buffer) const;

  private:
    std::unique_ptr<index_dense_t> index_;
};

std::unique_ptr<NativeIndex> new_native_index(IndexOptions const& options);
