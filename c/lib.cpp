#include "../src/index_punned_dense.hpp"

extern "C" {
#include "usearch.h"
}

using namespace unum::usearch;
using namespace unum;

using label_t = int;
using distance_t = float;
using punned_index_t = index_punned_dense_gt<label_t>;
using span_t = span_gt<float>;

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
    case usearch_scalar_b1_k: return scalar_kind_t::b1_k;
    default: return scalar_kind_t::unknown_k;
    }
}

void usearch_init(usearch_init_options_t* options, usearch_index_t* index, usearch_error_t* error) {

    try {
        config_t config;
        config.expansion_add = static_cast<std::size_t>(expansion_add);
        config.expansion_search = static_cast<std::size_t>(expansion_search);
        config.connectivity = static_cast<std::size_t>(connectivity);
        config.max_elements = static_cast<std::size_t>(capacity);
        config.max_threads_add = std::thread::hardware_concurrency();
        config.max_threads_search = std::thread::hardware_concurrency();

        // TODO: Implement safe buffers for error messages, either here on in
        // the C++ type-punned implementation.
        // static constexpr std::size_t error_limit_k = 128;
        // struct safe_punned_index_t {
        //     punned_index_t punned_index;
        //     char error_buffers[1];
        // };
        // std::size_t bytes_for_error_messages = error_limit_k * config.max_threads();
        // std::size_t bytes_to_alloc = sizeof(punned_index_t) + bytes_for_error_messages;
        // safe_punned_index_t* result = (safe_punned_index_t*)std::malloc(bytes_to_alloc);

        scalar_kind_t accuracy = scalar_kind_from_name(accuracy_str, accuracy_len);
        metric_kind_t metric_kind = metric_from_name(metric_str, metric_len);
        punned_index_t index = make_punned<punned_index_t>( //
            metric_kind, static_cast<std::size_t>(dimensions), accuracy, config);

        punned_index_t* result_ptr = new punned_index_t(std::move(index));
        return result_ptr;
    } catch (std::exception const&) {
    }
    return NULL;
}

void usearch_free(punned_index_t* index) { delete index; }

char const* usearch_save(punned_index_t* index, char const* path) {
    try {
        index->save(path);
        return NULL;
    } catch (std::exception& e) {
        // q:: added these to make sure I do not pass a potentially dangling e.what()
        // though I realize allocation below may fail and I am not checking for it.
        // what can I do there? Golang side tries to free whatever string is passed from
        // so passing a compile time constant "double fault" string is not an option
        char* res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

char const* usearch_load(punned_index_t* index, char const* path) {
    try {
        index->load(path);
        return NULL;
    } catch (std::exception& e) {
        char* res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

char const* usearch_view(punned_index_t* index, char const* path) {
    try {
        index->view(path);
        return NULL;
    } catch (std::exception& e) {
        char* res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

int usearch_size(punned_index_t* index) { return index->size(); }
int usearch_connectivity(punned_index_t* index) { return index->connectivity(); }
int usearch_dimensions(punned_index_t* index) { return index->dimensions(); }
int usearch_capacity(punned_index_t* index) { return index->capacity(); }

char const* usearch_reserve(punned_index_t* index, int capacity) {
    try {
        index->reserve(capacity);
        return NULL;
    } catch (std::exception& e) {
        char* res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

char const* usearch_add(punned_index_t* index, int label, float* vector) {
    // q:: I followed the java example to have try catches everywhere
    // but they are kind of useless as most errors are outside of cpp so
    // those translate into segfaults and are not caught by the runtime
    try {
        index->add(label, vector);
    } catch (std::exception& e) {
        char* res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
    return NULL;
}

SearchResults usearch_search(punned_index_t* index, float* query, int query_len, int limit) {
    // todo:: this could be allocated as golang slice
    // to avoid a copy. not sure how it interacts with gc
    // that is why doing this now
    label_t* matches_data = new label_t[limit];

    // todo:: pass the distances to outside world
    distance_t* dist_data = new distance_t[limit];
    SearchResults res{0};
    try {
        span_t vector_span = span_t{query, static_cast<std::size_t>(query_len)};
        res.Len = index->search(vector_span, static_cast<std::size_t>(limit)).dump_to(matches_data, dist_data);
    } catch (std::exception& e) {
        res.Error = new char[strlen(e.what()) + 1];
        strcpy(res.Error, e.what());
        // res.Error = res;
    }
    res.Labels = matches_data;
    res.Distances = dist_data;
    return res;
}
