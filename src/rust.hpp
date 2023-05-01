#pragma once
#include "rust/cxx.h"

#include "advanced.hpp"

struct Matches;

enum class Metric;

enum class Accuracy;

class Index {
  public:
    using label_t = std::uint32_t;
    using distance_t = unum::usearch::punned_distance_t;
    using native_index_t = unum::usearch::auto_index_gt<label_t>;

    Index(std::shared_ptr<native_index_t> index);

    void reserve(size_t) const;

    void add(uint32_t label, rust::Slice<float const> vector) const;
    void add_in_thread(uint32_t label, rust::Slice<float const> vector, size_t thread) const;

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
    std::shared_ptr<native_index_t> index_;
};

std::unique_ptr<Index> new_ip(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search);
std::unique_ptr<Index> new_l2(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search);
std::unique_ptr<Index> new_cos(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search);
std::unique_ptr<Index> new_haversine(rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search);
