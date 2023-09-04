#include <cassert>

#include <usearch/index_dense.hpp>

extern "C" {
#include "usearch.h"
}

using namespace unum::usearch;
using namespace unum;

using add_result_t = typename index_dense_t::add_result_t;
using search_result_t = typename index_dense_t::search_result_t;
using labeling_result_t = typename index_dense_t::labeling_result_t;

static_assert(std::is_same<usearch_key_t, index_dense_t::key_t>::value, "Type mismatch between C and C++");
static_assert(std::is_same<usearch_distance_t, index_dense_t::distance_t>::value, "Type mismatch between C and C++");

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
    case usearch_scalar_i8_k: return scalar_kind_t::i8_k;
    case usearch_scalar_b1_k: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

add_result_t add_(index_dense_t* index, usearch_key_t key, void const* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->add(key, (f32_t const*)vector);
    case scalar_kind_t::f64_k: return index->add(key, (f64_t const*)vector);
    case scalar_kind_t::f16_k: return index->add(key, (f16_t const*)vector);
    case scalar_kind_t::i8_k: return index->add(key, (i8_t const*)vector);
    case scalar_kind_t::b1x8_k: return index->add(key, (b1x8_t const*)vector);
    default: return add_result_t{}.failed("Unknown scalar kind!");
    }
}

std::size_t get_(index_dense_t* index, usearch_key_t key, size_t count, void* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->get(key, (f32_t*)vector, count);
    case scalar_kind_t::f64_k: return index->get(key, (f64_t*)vector, count);
    case scalar_kind_t::f16_k: return index->get(key, (f16_t*)vector, count);
    case scalar_kind_t::i8_k: return index->get(key, (i8_t*)vector, count);
    case scalar_kind_t::b1x8_k: return index->get(key, (b1x8_t*)vector, count);
    default: return search_result_t().failed("Unknown scalar kind!");
    }
}

search_result_t search_(index_dense_t* index, void const* vector, scalar_kind_t kind, size_t n) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->search((f32_t const*)vector, n);
    case scalar_kind_t::f64_k: return index->search((f64_t const*)vector, n);
    case scalar_kind_t::f16_k: return index->search((f16_t const*)vector, n);
    case scalar_kind_t::i8_k: return index->search((i8_t const*)vector, n);
    case scalar_kind_t::b1x8_k: return index->search((b1x8_t const*)vector, n);
    default: return search_result_t().failed("Unknown scalar kind!");
    }
}

extern "C" {

USEARCH_EXPORT usearch_index_t usearch_init(usearch_init_options_t* options, usearch_error_t* error) {

    assert(options && error);

    index_dense_config_t config(options->connectivity, options->expansion_add, options->expansion_search);
    config.multi = options->multi;
    metric_kind_t metric_kind = to_native_metric(options->metric_kind);
    scalar_kind_t scalar_kind = to_native_scalar(options->quantization);

    auto metric_lambda = [options](byte_t const* a, byte_t const* b) -> usearch_distance_t {
        return options->metric((void const*)a, (void const*)b);
    };
    metric_punned_t metric = //
        options->metric ? metric_punned_t(metric_lambda, options->dimensions, metric_kind, scalar_kind)
                        : metric_punned_t(options->dimensions, metric_kind, scalar_kind);

    index_dense_t index = index_dense_t::make(metric, config);
    index_dense_t* result_ptr = new index_dense_t(std::move(index));
    return result_ptr;
}

USEARCH_EXPORT void usearch_free(usearch_index_t index, usearch_error_t*) {
    delete reinterpret_cast<index_dense_t*>(index);
}

USEARCH_EXPORT size_t usearch_serialized_length(usearch_index_t index, usearch_error_t*) {
    assert(index);
    return reinterpret_cast<index_dense_t*>(index)->serialized_length();
}

USEARCH_EXPORT void usearch_save(usearch_index_t index, char const* path, usearch_error_t* error) {

    assert(index && path && error);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->save(path);
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT void usearch_load(usearch_index_t index, char const* path, usearch_error_t* error) {

    assert(index && path && error);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->load(path);
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT void usearch_view(usearch_index_t index, char const* path, usearch_error_t* error) {

    assert(index && path && error);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->view(path);
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT void usearch_save_buffer(usearch_index_t index, void* buffer, size_t length, usearch_error_t* error) {

    assert(index && buffer && length && error);
    memory_mapped_file_t memory_map((byte_t*)buffer, length);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->save(std::move(memory_map));
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT void usearch_load_buffer(usearch_index_t index, void const* buffer, size_t length,
                                        usearch_error_t* error) {

    assert(index && buffer && length && error);
    memory_mapped_file_t memory_map((byte_t*)buffer, length);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->load(std::move(memory_map));
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT void usearch_view_buffer(usearch_index_t index, void const* buffer, size_t length,
                                        usearch_error_t* error) {

    assert(index && buffer && length && error);
    memory_mapped_file_t memory_map((byte_t*)buffer, length);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->view(std::move(memory_map));
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT size_t usearch_size(usearch_index_t index, usearch_error_t*) { //
    return reinterpret_cast<index_dense_t*>(index)->size();
}

USEARCH_EXPORT size_t usearch_capacity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_dense_t*>(index)->capacity();
}

USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_dense_t*>(index)->dimensions();
}

USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_dense_t*>(index)->connectivity();
}

USEARCH_EXPORT void usearch_reserve(usearch_index_t index, size_t capacity, usearch_error_t* error) {
    assert(index && error);
    if (!reinterpret_cast<index_dense_t*>(index)->reserve(capacity))
        *error = "Out of memory!";
}

USEARCH_EXPORT void usearch_add(                                                              //
    usearch_index_t index, usearch_key_t key, void const* vector, usearch_scalar_kind_t kind, //
    usearch_error_t* error) {

    assert(index && vector && error);
    add_result_t result = add_(reinterpret_cast<index_dense_t*>(index), key, vector, to_native_scalar(kind));
    if (!result)
        *error = result.error.release();
}

USEARCH_EXPORT bool usearch_contains(usearch_index_t index, usearch_key_t key, usearch_error_t*) {
    assert(index);
    return reinterpret_cast<index_dense_t*>(index)->contains(key);
}

USEARCH_EXPORT size_t usearch_count(usearch_index_t index, usearch_key_t key, usearch_error_t*) {
    assert(index);
    return reinterpret_cast<index_dense_t*>(index)->count(key);
}

USEARCH_EXPORT size_t usearch_search(                                                            //
    usearch_index_t index, void const* vector, usearch_scalar_kind_t kind, size_t results_limit, //
    usearch_key_t* found_keys, usearch_distance_t* found_distances, usearch_error_t* error) {

    assert(index && vector && error);
    search_result_t result =
        search_(reinterpret_cast<index_dense_t*>(index), vector, to_native_scalar(kind), results_limit);
    if (!result) {
        *error = result.error.release();
        return 0;
    }

    return result.dump_to(found_keys, found_distances);
}

USEARCH_EXPORT size_t usearch_get(                          //
    usearch_index_t index, usearch_key_t key, size_t count, //
    void* vectors, usearch_scalar_kind_t kind, usearch_error_t*) {

    assert(index && vectors);
    return get_(reinterpret_cast<index_dense_t*>(index), key, count, vectors, to_native_scalar(kind));
}

USEARCH_EXPORT size_t usearch_remove(usearch_index_t index, usearch_key_t key, usearch_error_t* error) {

    assert(index && error);
    labeling_result_t result = reinterpret_cast<index_dense_t*>(index)->remove(key);
    if (!result)
        *error = result.error.release();
    return result.completed;
}

USEARCH_EXPORT size_t usearch_rename(usearch_index_t index, usearch_key_t from, usearch_key_t to,
                                     usearch_error_t* error) {

    assert(index && error);
    labeling_result_t result = reinterpret_cast<index_dense_t*>(index)->rename(from, to);
    if (!result)
        *error = result.error.release();
    return result.completed;
}
}
