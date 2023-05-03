#include "../src/advanced.hpp"
#include <algorithm>

extern "C" {
#include "golang.h"
}

using namespace unum::usearch;
using namespace unum;

// todo:: pretty sure use of int here is not portible
using label_t = int;
using distance_t = float;
using native_index_t = auto_index_gt<label_t>;
using span_t = span_gt<float>;

extern "C" {
typedef struct config_t Config;

int Bench(char* vectors, char* queries, char* neighbors) {
    std::printf("benchmarking");
    return 0;
}

native_index_t* new_index(char* metric_str, int metric_len, char* accuracy_str, int accuracy_len, int dimensions,
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
}

void destroy(native_index_t* index) { delete index; }

const char* save(native_index_t* index, char* path) {
    try {
        index->save(path);
    } catch (std::exception& e) {
        return e.what();
    }
}

const char* load(native_index_t* index, char* path) {
    try {
        index->load(path);
    } catch (std::exception& e) {
        return e.what();
    }
}

const char* view(native_index_t* index, char* path) {
    try {
        index->view(path);
    } catch (std::exception& e) {
        return e.what();
    }
}

int size(native_index_t* index) { return index->size(); }

int connectivity(native_index_t* index) { return index->connectivity(); }

// q:: why is this not "dimension"?
int dimensions(native_index_t* index) { return index->dimensions(); }
int capacity(native_index_t* index) { return index->capacity(); }

const char* set_capacity(native_index_t* index, int capacity) {
    try {
        index->reserve(capacity);
    } catch (std::exception& e) {
        // todo:: q:: does e need to free'd?
        return e.what();
    }
}

const char* add(native_index_t* index, int label, float* vector) {
    // q:: I followed the java example to have try catches everywhere
    // but they are kind of useless as most errors areoutside of cpp so
    // translate into sefaults and are not caught by the runtime
    try {
        // todo:: the code cannot currently be used in a multithreaded environment
        // because of golang's roaming gorountines, but once we have a way to
        // work around it, we should revisit this as this check has a toctou race
        if (index->size() + 1 >= index->capacity())
            index->reserve(ceil2(index->size() + 1));
        index->add(label, vector);
    } catch (std::exception& e) {
        return e.what();
    }
    return NULL;
}

SearchResults search(native_index_t* index, float* query, int query_len, int limit) {
    // todo:: this could be allocated as golang slice
    // to avoid a copy. not sure how it interacts with gc
    // that is why doing this now
    label_t* matches_data = new label_t[limit];
    SearchResults res{0};
    try {
        span_t vector_span = span_t{query, static_cast<std::size_t>(query_len)};
        res.LabelsLen = index->search( //
            vector_span, static_cast<std::size_t>(limit), matches_data, NULL);
    } catch (std::exception& e) {
        res.Error = e.what();
    }
    res.Labels = matches_data;
    return res;
}
}