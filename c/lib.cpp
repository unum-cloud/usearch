#include "../src/punned.hpp"

extern "C" {
#include "usearch.h"
}

using namespace unum::usearch;
using namespace unum;

// todo:: pretty sure use of int here is not portible
using label_t = int;
using distance_t = float;
using native_index_t = punned_gt<label_t>;
using span_t = span_gt<float>;

extern "C" {
typedef struct config_t Config;

native_index_t* usearch_new_index(char* metric_str, int metric_len, char* accuracy_str, int accuracy_len, int dimensions,
                          int capacity, int connectivity, int expansion_add, int expansion_search) {
    try {
        config_t config;
        config.expansion_add = static_cast<std::size_t>(expansion_add);
        config.expansion_search = static_cast<std::size_t>(expansion_search);
        config.connectivity = static_cast<std::size_t>(connectivity);
        config.max_elements = static_cast<std::size_t>(capacity);
        // todo:: what should those be?
        config.max_threads_add = std::thread::hardware_concurrency();
        config.max_threads_search = std::thread::hardware_concurrency();

        accuracy_t accuracy = accuracy_from_name(accuracy_str, accuracy_len);
        native_index_t index = index_from_name<native_index_t>( //
            metric_str, metric_len, static_cast<std::size_t>(dimensions), accuracy, config);

        native_index_t* result_ptr = new native_index_t(std::move(index));
        return result_ptr;
    } catch (std::exception& e) {
        printf("error %s\n", e.what());
    }
    return NULL;
}

void usearch_destroy(native_index_t* index) { delete index; }

const char* usearch_save(native_index_t* index, char* path) {
    try {
        index->save(path);
        return NULL;
    } catch (std::exception& e) {
        //q:: added these to make sure I do not pass a potentially dangling e.what()
        // though I realize allocation below may fail and I am not checking for it.
        // what can I do there? Golang side tries to free whatever string is passed from
        // so passing a compile time constant "double fault" string is not an option
        char *res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

const char* usearch_load(native_index_t* index, char* path) {
    try {
        index->load(path);
        return NULL;
    } catch (std::exception& e) {
        char *res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

const char* usearch_view(native_index_t* index, char* path) {
    try {
        index->view(path);
        return NULL;
    } catch (std::exception& e) {
        char *res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

int usearch_size(native_index_t* index) { return index->size(); }
int usearch_connectivity(native_index_t* index) { return index->connectivity(); }
int usearch_dimensions(native_index_t* index) { return index->dimensions(); }
int usearch_capacity(native_index_t* index) { return index->capacity(); }

const char* usearch_reserve(native_index_t* index, int capacity) {
    try {
        index->reserve(capacity);
        return NULL;
    } catch (std::exception& e) {
        char *res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;
    }
}

const char* usearch_add(native_index_t* index, int label, float* vector) {
    // q:: I followed the java example to have try catches everywhere
    // but they are kind of useless as most errors are outside of cpp so
    // those translate into segaults and are not caught by the runtime
    try {
        index->add(label, vector);
    } catch (std::exception& e) {
        char *res = new char[strlen(e.what()) + 1];
        strcpy(res, e.what());
        return res;    }
    return NULL;
}

SearchResults usearch_search(native_index_t* index, float* query, int query_len, int limit) {
    // todo:: this could be allocated as golang slice
    // to avoid a copy. not sure how it interacts with gc
    // that is why doing this now
    label_t* matches_data = new label_t[limit];

    // todo:: pass the distances to outside world
    distance_t* dist_data = new distance_t[limit];
    SearchResults res{0};
    try {
        span_t vector_span = span_t{query, static_cast<std::size_t>(query_len)};
        res.Len = index->search( //
            vector_span, static_cast<std::size_t>(limit)).dump_to(matches_data, dist_data);
    } catch (std::exception& e) {
        res.Error = new char[strlen(e.what()) + 1];
        strcpy(res.Error, e.what());
        // res.Error = res;
    }
    res.Labels = matches_data;
    res.Distances = dist_data;
    return res;
}
}
