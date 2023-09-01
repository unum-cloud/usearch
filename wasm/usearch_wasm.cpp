#include <lib.cpp>

#ifdef USEARCH_USE_EMSCRIPTEN
#include <emscripten/emscripten.h>
#define USEARCH_EXPORT EMSCRIPTEN_KEEPALIVE
#endif // USEARCH_USE_EMSCRIPTEN

#include "usearch-wasm.h"

#pragma region - Results and error converters
void convert_err(usearch_error_t usearch_error, usearch_wasm_error_t& wasm_error) {
    wasm_error.ptr = const_cast<char*>(usearch_error);
    wasm_error.len = strlen(usearch_error);
}

void convert_err_with_res(usearch_error_t usearch_error, usearch_index_t result,
                          usearch_wasm_expected_index_error_t* error) {
    if (usearch_error) {
        error->is_err = true;
        convert_err(usearch_error, error->val.err);
    } else {
        error->is_err = false;
        error->val.ok = reinterpret_cast<usearch_wasm_index_t>(result);
    }
}

void convert_err_with_res(usearch_error_t usearch_error, size_t result, usearch_wasm_expected_size_error_t* error) {
    if (usearch_error) {
        error->is_err = true;
        convert_err(usearch_error, error->val.err);
    } else {
        error->is_err = false;
        error->val.ok = static_cast<usearch_wasm_size_t>(result);
    }
}

void convert_err_with_res(usearch_error_t usearch_error, bool result, usearch_wasm_expected_bool_error_t* error) {
    if (usearch_error) {
        error->is_err = true;
        convert_err(usearch_error, error->val.err);
    } else {
        error->is_err = false;
        error->val.ok = result;
    }
}
#pragma endregion - Results and error converters

void usearch_wasm_init(usearch_wasm_init_options_t* options, usearch_wasm_expected_index_error_t* error) {
    usearch_init_options_t opts;
    opts.metric_kind = static_cast<usearch_metric_kind_t>(options->metric_kind);
    opts.metric = NULL;
    opts.quantization = static_cast<usearch_scalar_kind_t>(options->quantization);
    opts.connectivity = static_cast<size_t>(options->connectivity);
    opts.dimensions = static_cast<size_t>(options->dimensions);
    opts.expansion_add = static_cast<size_t>(options->expansion_add);
    opts.expansion_search = static_cast<size_t>(options->expansion_search);

    usearch_error_t usearch_error = nullptr;
    usearch_index_t result = usearch_init(&opts, &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_release(usearch_wasm_index_t index, usearch_wasm_error_t* error) {
    usearch_free(reinterpret_cast<usearch_index_t>(index), reinterpret_cast<usearch_error_t*>(error->ptr));
}

void usearch_wasm_save(usearch_wasm_index_t index, usearch_wasm_string_t* path, usearch_wasm_error_t* error) {
    usearch_save(reinterpret_cast<usearch_index_t>(index), path->ptr, reinterpret_cast<usearch_error_t*>(error->ptr));
}

void usearch_wasm_load(usearch_wasm_index_t index, usearch_wasm_string_t* path, usearch_wasm_error_t* error) {
    usearch_load(reinterpret_cast<usearch_index_t>(index), path->ptr, reinterpret_cast<usearch_error_t*>(error->ptr));
}

void usearch_wasm_view(usearch_wasm_index_t index, usearch_wasm_string_t* path, usearch_wasm_error_t* error) {
    usearch_view(reinterpret_cast<usearch_index_t>(index), path->ptr, reinterpret_cast<usearch_error_t*>(error->ptr));
}

void usearch_wasm_size(usearch_wasm_index_t index, usearch_wasm_expected_size_error_t* error) {
    usearch_error_t usearch_error = nullptr;
    size_t result = usearch_size(reinterpret_cast<usearch_index_t>(index), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_capacity(usearch_wasm_index_t index, usearch_wasm_expected_size_error_t* error) {
    usearch_error_t usearch_error = nullptr;
    size_t result = usearch_capacity(reinterpret_cast<usearch_index_t>(index), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_dimensions(usearch_wasm_index_t index, usearch_wasm_expected_size_error_t* error) {
    usearch_error_t usearch_error = nullptr;
    size_t result = usearch_dimensions(reinterpret_cast<usearch_index_t>(index), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_connectivity(usearch_wasm_index_t index, usearch_wasm_expected_size_error_t* error) {
    usearch_error_t usearch_error = nullptr;
    size_t result = usearch_connectivity(reinterpret_cast<usearch_index_t>(index), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_reserve(usearch_wasm_index_t index, usearch_wasm_size_t capacity, usearch_wasm_error_t* error) {
    usearch_reserve(reinterpret_cast<usearch_index_t>(index), static_cast<size_t>(capacity),
                    reinterpret_cast<usearch_error_t*>(error->ptr));
}

void usearch_wasm_add(usearch_wasm_index_t index, usearch_wasm_key_t key, usearch_wasm_vector_t* array,
                      usearch_wasm_scalar_kind_t vector_kind, usearch_wasm_error_t* error) {
    usearch_add(reinterpret_cast<usearch_index_t>(index), static_cast<usearch_key_t>(key),
                reinterpret_cast<void*>(array->val.bytes.ptr), static_cast<usearch_scalar_kind_t>(vector_kind),
                reinterpret_cast<usearch_error_t*>(error->ptr));
}

void usearch_wasm_contains(usearch_wasm_index_t index, usearch_wasm_key_t key,
                           usearch_wasm_expected_bool_error_t* error) {

    usearch_error_t usearch_error = nullptr;
    bool result =
        usearch_contains(reinterpret_cast<usearch_index_t>(index), static_cast<usearch_key_t>(key), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_search(usearch_wasm_index_t index, usearch_wasm_vector_t* array, usearch_wasm_scalar_kind_t kind,
                         usearch_wasm_size_t results_limit, usearch_wasm_keys_t* found_labels,
                         usearch_wasm_distances_t* found_distances, usearch_wasm_expected_size_error_t* error) {

    usearch_error_t usearch_error = nullptr;
    size_t result =
        usearch_search(reinterpret_cast<usearch_index_t>(index), reinterpret_cast<void*>(array->val.bytes.ptr),
                       static_cast<usearch_scalar_kind_t>(kind), static_cast<size_t>(results_limit),
                       reinterpret_cast<usearch_key_t*>(found_labels->ptr),
                       static_cast<usearch_distance_t*>(found_distances->ptr), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_get(usearch_wasm_index_t index, usearch_wasm_key_t key, usearch_wasm_vector_t* array,
                      usearch_wasm_scalar_kind_t vector_kind, usearch_wasm_expected_bool_error_t* error) {

    usearch_error_t usearch_error = nullptr;
    bool result = usearch_get(reinterpret_cast<usearch_index_t>(index), static_cast<usearch_key_t>(key),
                              reinterpret_cast<void*>(array->val.bytes.ptr),
                              static_cast<usearch_scalar_kind_t>(vector_kind), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}

void usearch_wasm_remove(usearch_wasm_index_t index, usearch_wasm_key_t key,
                         usearch_wasm_expected_bool_error_t* error) {

    usearch_error_t usearch_error = nullptr;
    bool result =
        usearch_remove(reinterpret_cast<usearch_index_t>(index), static_cast<usearch_key_t>(key), &usearch_error);
    convert_err_with_res(usearch_error, result, error);
}
