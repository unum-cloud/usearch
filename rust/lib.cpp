#include "lib.hpp"
#include "usearch/rust/lib.rs.h"

using namespace unum::usearch;
using namespace unum;

using punned_t = typename Index::punned_t;
using add_result_t = typename punned_t::add_result_t;
using search_result_t = typename punned_t::search_result_t;
using serialization_result_t = typename punned_t::serialization_result_t;

Index::Index(std::unique_ptr<punned_t> index) : index_(std::move(index)) {}

void Index::add_in_thread(uint32_t label, rust::Slice<float const> vector, size_t thread) const {
    add_config_t config;
    config.thread = thread;
    index_->add(label, vector.data(), config).error.raise();
}

Matches Index::search_in_thread(rust::Slice<float const> vector, size_t count, size_t thread) const {
    Matches matches;
    matches.labels.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.labels.push_back(0), matches.distances.push_back(0);
    search_config_t config;
    config.thread = thread;
    search_result_t result = index_->search(vector.data(), count, config);
    result.error.raise();
    matches.count = result.dump_to(matches.labels.data(), matches.distances.data());
    matches.labels.truncate(matches.count);
    matches.distances.truncate(matches.count);
    return matches;
}

void Index::add(uint32_t label, rust::Slice<float const> vector) const {
    index_->add(label, vector.data()).error.raise();
}

Matches Index::search(rust::Slice<float const> vector, size_t count) const {
    Matches matches;
    matches.labels.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.labels.push_back(0), matches.distances.push_back(0);
    search_result_t result = index_->search(vector.data(), count);
    result.error.raise();
    matches.count = result.dump_to(matches.labels.data(), matches.distances.data());
    matches.labels.truncate(matches.count);
    matches.distances.truncate(matches.count);
    return matches;
}

void Index::reserve(size_t capacity) const { index_->reserve(capacity); }

size_t Index::dimensions() const { return index_->dimensions(); }
size_t Index::connectivity() const { return index_->connectivity(); }
size_t Index::size() const { return index_->size(); }
size_t Index::capacity() const { return index_->capacity(); }

void Index::save(rust::Str path) const { index_->save(std::string(path).c_str()); }
void Index::load(rust::Str path) const { index_->load(std::string(path).c_str()); }
void Index::view(rust::Str path) const { index_->view(std::string(path).c_str()); }

accuracy_t accuracy(rust::Str quant) { return accuracy_from_name(quant.data(), quant.size()); }

std::unique_ptr<Index> wrap(punned_t&& index) {
    std::unique_ptr<punned_t> punned_ptr;
    punned_ptr.reset(new punned_t(std::move(index)));
    std::unique_ptr<Index> result;
    result.reset(new Index(std::move(punned_ptr)));
    return result;
}

index_config_t config(size_t connectivity, size_t exp_add, size_t exp_search) {
    index_config_t result;
    result.connectivity = connectivity ? connectivity : default_connectivity();
    result.expansion_add = exp_add ? exp_add : default_expansion_add();
    result.expansion_search = exp_search ? exp_search : default_expansion_search();
    return result;
}

std::unique_ptr<Index> new_ip(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(punned_t::ip(dims, accuracy(quant), config(connectivity, exp_add, exp_search)));
}

std::unique_ptr<Index> new_l2sq(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(punned_t::l2sq(dims, accuracy(quant), config(connectivity, exp_add, exp_search)));
}

std::unique_ptr<Index> new_cos(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(punned_t::cos(dims, accuracy(quant), config(connectivity, exp_add, exp_search)));
}

std::unique_ptr<Index> new_haversine(rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(punned_t::haversine(accuracy(quant), config(connectivity, exp_add, exp_search)));
}
