#include "lib.hpp"
#include "usearch/rust/lib.rs.h"

using namespace unum::usearch;
using namespace unum;

using index_t = typename Index::index_t;
using add_result_t = typename index_t::add_result_t;
using search_result_t = typename index_t::search_result_t;

Index::Index(std::unique_ptr<index_t> index) : index_(std::move(index)) {}

void Index::add_in_thread(key_t key, rust::Slice<float const> vector, size_t thread) const {
    index_add_config_t config;
    config.thread = thread;
    config.expansion = index_->expansion_add();
    index_->add(key, vector.data(), config).error.raise();
}

Matches Index::search_in_thread(rust::Slice<float const> vector, size_t count, size_t thread) const {
    Matches matches;
    matches.keys.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.keys.push_back(0), matches.distances.push_back(0);
    index_search_config_t config;
    config.thread = thread;
    config.expansion = index_->expansion_search();
    search_result_t result = index_->search(vector.data(), count, config);
    result.error.raise();
    matches.count = result.dump_to(matches.keys.data(), matches.distances.data());
    matches.keys.truncate(matches.count);
    matches.distances.truncate(matches.count);
    return matches;
}

void Index::add(key_t key, rust::Slice<float const> vector) const { index_->add(key, vector.data()).error.raise(); }

Matches Index::search(rust::Slice<float const> vector, size_t count) const {
    Matches matches;
    matches.keys.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.keys.push_back(0), matches.distances.push_back(0);
    search_result_t result = index_->search(vector.data(), count);
    result.error.raise();
    matches.count = result.dump_to(matches.keys.data(), matches.distances.data());
    matches.keys.truncate(matches.count);
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

scalar_kind_t quantization(rust::Str quant) { return scalar_kind_from_name(quant.data(), quant.size()); }

std::unique_ptr<Index> wrap(index_t&& index) {
    std::unique_ptr<index_t> punned_ptr;
    punned_ptr.reset(new index_t(std::move(index)));
    std::unique_ptr<Index> result;
    result.reset(new Index(std::move(punned_ptr)));
    return result;
}

metric_kind_t rust_to_cpp_metric(MetricKind value) {
    switch (value) {
    case MetricKind::IP: return metric_kind_t::ip_k;
    case MetricKind::L2Sq: return metric_kind_t::l2sq_k;
    case MetricKind::Cos: return metric_kind_t::cos_k;
    case MetricKind::Pearson: return metric_kind_t::pearson_k;
    case MetricKind::Haversine: return metric_kind_t::haversine_k;
    case MetricKind::Hamming: return metric_kind_t::hamming_k;
    case MetricKind::Tanimoto: return metric_kind_t::tanimoto_k;
    case MetricKind::Sorensen: return metric_kind_t::sorensen_k;
    default: return metric_kind_t::unknown_k;
    }
}

scalar_kind_t rust_to_cpp_scalar(ScalarKind value) {
    switch (value) {
    case ScalarKind::F8: return scalar_kind_t::f8_k;
    case ScalarKind::F16: return scalar_kind_t::f16_k;
    case ScalarKind::F32: return scalar_kind_t::f32_k;
    case ScalarKind::F64: return scalar_kind_t::f64_k;
    case ScalarKind::B1: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

std::unique_ptr<Index> new_index(IndexOptions const& options) {
    metric_kind_t metric_kind = rust_to_cpp_metric(options.metric);
    scalar_kind_t scalar_kind = rust_to_cpp_scalar(options.quantization);
    metric_punned_t metric(options.dimensions, metric_kind, scalar_kind);
    index_dense_config_t config(options.connectivity, options.expansion_add, options.expansion_search);
    return wrap(index_t::make(metric, config));
}
