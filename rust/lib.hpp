#pragma once
#include "rust/cxx.h"

#include <memory> // `std::shared_ptr`

#include <usearch/index_dense.hpp>

struct Matches;
struct IndexOptions;

class Index {
  public:
    using metric_t = unum::usearch::metric_punned_t;
    using distance_t = unum::usearch::distance_punned_t;
    using index_dense_t = unum::usearch::index_dense_t;
    using add_result_t = typename index_dense_t::add_result_t;
    using search_result_t = typename index_dense_t::search_result_t;
    using key_t = typename index_dense_t::key_t;

    Index(std::unique_ptr<index_dense_t> index);

    void reserve(size_t) const;

    void add(key_t key, rust::Slice<float const> vector) const;
    void add_i8(key_t key, rust::Slice<int8_t const> vector) const;
    void add_f16(key_t key, rust::Slice<uint16_t const> vector) const;
    void add_f32(key_t key, rust::Slice<float const> vector) const;
    void add_f64(key_t key, rust::Slice<double const> vector) const;

    Matches search(rust::Slice<float const> vector, size_t count) const;
    Matches search_i8(rust::Slice<int8_t const> vector, size_t count) const;
    Matches search_f16(rust::Slice<uint16_t const> vector, size_t count) const;
    Matches search_f32(rust::Slice<float const> vector, size_t count) const;
    Matches search_f64(rust::Slice<double const> vector, size_t count) const;

    bool remove(key_t key) const;
    bool contains(key_t key) const;
    bool rename(key_t from, key_t to) const;

    size_t dimensions() const;
    size_t connectivity() const;
    size_t size() const;
    size_t capacity() const;
    size_t serialized_length() const;

    void save(rust::Str path) const;
    void load(rust::Str path) const;
    void view(rust::Str path) const;

    void save_to_buffer(rust::Slice<uint8_t> buffer) const;
    void load_from_buffer(rust::Slice<uint8_t const> buffer) const;
    void view_from_buffer(rust::Slice<uint8_t const> buffer) const;

  private:
    std::unique_ptr<index_dense_t> index_;
};

std::unique_ptr<Index> new_index(IndexOptions const& options);
