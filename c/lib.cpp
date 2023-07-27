#include <cassert>

#include <usearch/index_dense.hpp>

#ifndef USEARCH_EXPORT
#define USEARCH_EXPORT
#endif

extern "C" {
#include "usearch.h"
}

using namespace unum::usearch;
using namespace unum;

using index_t = index_dense_t;
using key_t = typename index_t::key_t;
using distance_t = typename index_t::distance_t;
using add_result_t = typename index_t::add_result_t;
using search_result_t = typename index_t::search_result_t;
using labeling_result_t = typename index_t::labeling_result_t;
using vector_view_t = span_gt<float>;

static_assert(std::is_same<usearch_key_t, key_t>::value, "Type mismatch between C and C++");

// helper functions that are not part of the C ABI
metric_kind_t to_native_metric(usearch_metric_kind_t kind) {
    switch (kind) {
    case usearch_metric_ip_k: return metric_kind_t::ip_k;
    case usearch_metric_l2sq_k: return metric_kind_t::l2sq_k;
    case usearch_metric_cos_k: return metric_kind_t::cos_k;
    case usearch_metric_haversine_k: return metric_kind_t::haversine_k;
    case usearch_metric_pearson_k: return metric_kind_t::pearson_k;
    case usearch_metric_jaccard_k: return metric_kind_t::jaccard_k;
    case usearch_metric_hamming_k: return metric_kind_t::hamming_k;
    case usearch_metric_tanimoto_k: return metric_kind_t::tanimoto_k;
    case usearch_metric_sorensen_k: return metric_kind_t::sorensen_k;
    default: return metric_kind_t::unknown_k;
    }
}

scalar_kind_t to_native_scalar(usearch_scalar_kind_t kind) {
    switch (kind) {
    case usearch_scalar_f32_k: return scalar_kind_t::f32_k;
    case usearch_scalar_f64_k: return scalar_kind_t::f64_k;
    case usearch_scalar_f16_k: return scalar_kind_t::f16_k;
    case usearch_scalar_f8_k: return scalar_kind_t::f8_k;
    case usearch_scalar_b1_k: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

add_result_t add_(index_t* index, usearch_key_t key, void const* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->add(key, (f32_t const*)vector);
    case scalar_kind_t::f64_k: return index->add(key, (f64_t const*)vector);
    case scalar_kind_t::f16_k: return index->add(key, (f16_t const*)vector);
    case scalar_kind_t::f8_k: return index->add(key, (f8_bits_t const*)vector);
    case scalar_kind_t::b1x8_k: return index->add(key, (b1x8_t const*)vector);
    default: return add_result_t{}.failed("Unknown scalar kind!");
    }
}

bool get_(index_t* index, key_t key, void* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->get(key, (f32_t*)vector);
    case scalar_kind_t::f64_k: return index->get(key, (f64_t*)vector);
    case scalar_kind_t::f16_k: return index->get(key, (f16_t*)vector);
    case scalar_kind_t::f8_k: return index->get(key, (f8_bits_t*)vector);
    case scalar_kind_t::b1x8_k: return index->get(key, (b1x8_t*)vector);
    default: return search_result_t().failed("Unknown scalar kind!");
    }
}

search_result_t search_(index_t* index, void const* vector, scalar_kind_t kind, size_t n) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->search((f32_t const*)vector, n);
    case scalar_kind_t::f64_k: return index->search((f64_t const*)vector, n);
    case scalar_kind_t::f16_k: return index->search((f16_t const*)vector, n);
    case scalar_kind_t::f8_k: return index->search((f8_bits_t const*)vector, n);
    case scalar_kind_t::b1x8_k: return index->search((b1x8_t const*)vector, n);
    default: return search_result_t().failed("Unknown scalar kind!");
    }
}

metric_punned_t udf(metric_kind_t kind, usearch_metric_t raw_ptr) {
    metric_punned_t result;
    result.kind_ = kind;
    result.func_ = [raw_ptr](span_punned_t a, span_punned_t b) -> distance_t {
        return raw_ptr((void const*)a.data(), (void const*)b.data());
    };
    return result;
}

extern "C" {

USEARCH_EXPORT usearch_index_t usearch_init(usearch_init_options_t* options, usearch_error_t* error) {

    assert(options && error);

    index_config_t config;
    config.connectivity = options->connectivity;
    index_t index =        //
        options->metric ?  //
            index_t::make( //
                options->dimensions, udf(to_native_metric(options->metric_kind), options->metric), config,
                to_native_scalar(options->quantization), options->expansion_add, options->expansion_search)
                        :  //
            index_t::make( //
                options->dimensions, to_native_metric(options->metric_kind), config,
                to_native_scalar(options->quantization), options->expansion_add, options->expansion_search);

    index_t* result_ptr = new index_t(std::move(index));
    return result_ptr;
}

USEARCH_EXPORT void usearch_free(usearch_index_t index, usearch_error_t*) { delete reinterpret_cast<index_t*>(index); }

USEARCH_EXPORT void usearch_save(usearch_index_t index, char const* path, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->save(path);
    if (!result)
        *error = result.error.what();
}

USEARCH_EXPORT void usearch_load(usearch_index_t index, char const* path, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->load(path);
    if (!result)
        *error = result.error.what();
}

USEARCH_EXPORT void usearch_view(usearch_index_t index, char const* path, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->view(path);
    if (!result)
        *error = result.error.what();
}

USEARCH_EXPORT size_t usearch_size(usearch_index_t index, usearch_error_t*) { //
    return reinterpret_cast<index_t*>(index)->size();
}

USEARCH_EXPORT size_t usearch_capacity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->capacity();
}

USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->dimensions();
}

USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->connectivity();
}

USEARCH_EXPORT void usearch_reserve(usearch_index_t index, size_t capacity, usearch_error_t*) {
    // TODO: Consider returning the new capacity.
    reinterpret_cast<index_t*>(index)->reserve(capacity);
}

USEARCH_EXPORT void usearch_add(                                                              //
    usearch_index_t index, usearch_key_t key, void const* vector, usearch_scalar_kind_t kind, //
    usearch_error_t* error) {
    add_result_t result = add_(reinterpret_cast<index_t*>(index), key, vector, to_native_scalar(kind));
    if (!result)
        *error = result.error.what();
}

USEARCH_EXPORT bool usearch_contains(usearch_index_t index, usearch_key_t key, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->contains(key);
}

USEARCH_EXPORT size_t usearch_search(                                                            //
    usearch_index_t index, void const* vector, usearch_scalar_kind_t kind, size_t results_limit, //
    usearch_key_t* found_labels, usearch_distance_t* found_distances, usearch_error_t* error) {
    search_result_t result = search_(reinterpret_cast<index_t*>(index), vector, to_native_scalar(kind), results_limit);
    if (!result) {
        *error = result.error.what();
        return 0;
    }

    return result.dump_to(found_labels, found_distances);
}

USEARCH_EXPORT bool usearch_get(              //
    usearch_index_t index, usearch_key_t key, //
    void* vector, usearch_scalar_kind_t kind, usearch_error_t*) {
    return get_(reinterpret_cast<index_t*>(index), key, vector, to_native_scalar(kind));
}

USEARCH_EXPORT bool usearch_remove(usearch_index_t index, usearch_key_t key, usearch_error_t* error) {
    labeling_result_t result = reinterpret_cast<index_t*>(index)->remove(key);
    if (!result)
        *error = result.error.what();
    return result.completed;
}
}
