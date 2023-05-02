#include "lib.hpp"
#include "usearch/rust/lib.rs.h"

using namespace unum::usearch;
using namespace unum;

using native_index_t = typename Index::native_index_t;

Index::Index(std::shared_ptr<native_index_t> index) : index_(index) {}

void Index::add_in_thread(uint32_t label, rust::Slice<float const> vector, size_t thread) const {
    index_->add(label, vector.data(), thread, true);
}

Matches Index::search_in_thread(rust::Slice<float const> vector, size_t count, size_t thread) const {
    Matches results;
    results.labels.reserve(count);
    results.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        results.labels.push_back(0), results.distances.push_back(0);
    results.count = index_->search(vector.data(), count, results.labels.data(), results.distances.data(), thread);
    results.labels.truncate(results.count);
    results.distances.truncate(results.count);
    return results;
}

void Index::add(uint32_t label, rust::Slice<float const> vector) const { return add_in_thread(label, vector, 0); }
Matches Index::search(rust::Slice<float const> vector, size_t count) const {
    return search_in_thread(vector, count, 0);
}

void Index::reserve(size_t capacity) const { index_->reserve(capacity); }

size_t Index::dimensions() const { return index_->dimensions(); }
size_t Index::connectivity() const { return index_->connectivity(); }
size_t Index::size() const { return index_->size(); }
size_t Index::capacity() const { return index_->capacity(); }

void Index::save(rust::Str path) const { index_->save(std::string(path).c_str()); }
void Index::load(rust::Str path) const { index_->load(std::string(path).c_str()); }
void Index::view(rust::Str path) const { index_->view(std::string(path).c_str()); }

accuracy_t accuracy(rust::Str quant) { return accuracy(quant.data(), quant.size()); }

std::unique_ptr<Index> wrap(native_index_t&& index) {
    std::shared_ptr<native_index_t> native = std::make_shared<native_index_t>(std::move(index));
    return std::unique_ptr<Index>(new Index(native));
}

config_t config(size_t connectivity, size_t exp_add, size_t exp_search) {
    config_t result;
    result.connectivity = connectivity ? connectivity : config_t::connectivity_default_k;
    result.expansion_add = exp_add ? exp_add : config_t::expansion_add_default_k;
    result.expansion_search = exp_search ? exp_search : config_t::expansion_search_default_k;
    return result;
}

std::unique_ptr<Index> new_ip(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(native_index_t::ip(dims, accuracy(quant), config(connectivity, exp_add, exp_search)));
}

std::unique_ptr<Index> new_l2(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(native_index_t::l2(dims, accuracy(quant), config(connectivity, exp_add, exp_search)));
}

std::unique_ptr<Index> new_cos(size_t dims, rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(native_index_t::cos(dims, accuracy(quant), config(connectivity, exp_add, exp_search)));
}

std::unique_ptr<Index> new_haversine(rust::Str quant, size_t connectivity, size_t exp_add, size_t exp_search) {
    return wrap(native_index_t::haversine(accuracy(quant), config(connectivity, exp_add, exp_search)));
}
