#pragma once
#include "rust/cxx.h"

#include <memory> // `std::shared_ptr`

#include <usearch/index_dense.hpp>

struct SearchResults;
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
    void add_in_thread(key_t key, rust::Slice<float const> vector, size_t thread) const;

    SearchResults search(rust::Slice<float const> vector, size_t count) const;
    SearchResults search_in_thread(rust::Slice<float const> vector, size_t count, size_t thread) const;

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
