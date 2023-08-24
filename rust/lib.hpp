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
    Matches search(rust::Slice<float const> vector, size_t count) const;

    bool remove(key_t key) const;
    bool contains(key_t key) const;
    bool rename(key_t from, key_t to) const;

    size_t dimensions() const;
    size_t connectivity() const;
    size_t size() const;
    size_t capacity() const;

    void save(rust::Str path) const;
    void load(rust::Str path) const;
    void view(rust::Str path) const;

  private:
    std::unique_ptr<index_dense_t> index_;
};

std::unique_ptr<Index> new_index(IndexOptions const& options);
