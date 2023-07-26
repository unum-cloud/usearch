#pragma once
#include "rust/cxx.h"

#include <memory> // `std::shared_ptr`

#include <usearch/index_dense.hpp>

struct Matches;
struct IndexOptions;

class Index {
  public:
    using metric_t = unum::usearch::index_dense_metric_t;
    using distance_t = unum::usearch::punned_distance_t;
    using index_t = unum::usearch::punned_small_t;
    using add_result_t = typename index_t::add_result_t;
    using search_result_t = typename index_t::search_result_t;
    using label_t = typename index_t::label_t;
    using id_t = typename index_t::id_t;

    Index(std::unique_ptr<index_t> index);

    void reserve(size_t) const;

    void add(label_t label, rust::Slice<float const> vector) const;
    void add_in_thread(label_t label, rust::Slice<float const> vector, size_t thread) const;

    Matches search(rust::Slice<float const> vector, size_t count) const;
    Matches search_in_thread(rust::Slice<float const> vector, size_t count, size_t thread) const;

    size_t dimensions() const;
    size_t connectivity() const;
    size_t size() const;
    size_t capacity() const;

    void save(rust::Str path) const;
    void load(rust::Str path) const;
    void view(rust::Str path) const;

  private:
    std::unique_ptr<index_t> index_;
};

std::unique_ptr<Index> new_index(IndexOptions const& options);
