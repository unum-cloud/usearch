#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "usearch.h"

#define ASSERT(must_be_true, message)                                                                                  \
    if (!(must_be_true)) {                                                                                             \
        printf("Assert: %s\n", message);                                                                               \
        exit(-1);                                                                                                      \
    }

float* create_vectors(size_t count, size_t dimensions) {
    float* data = (float*)malloc(count * dimensions * sizeof(float));
    ASSERT(data, "Failed to allocate memory");
    for (size_t idx = 0; idx < count * dimensions; ++idx)
        data[idx] = (float)rand() / (float)RAND_MAX;
    return data;
}

usearch_init_options_t create_options(size_t dimensions) {
    usearch_init_options_t opts;
    opts.connectivity = 2; // 32 in faiss
    opts.dimensions = dimensions;
    opts.expansion_add = 40;    // 40 in faiss
    opts.expansion_search = 16; // 10 in faiss
    opts.metric_kind = usearch_metric_ip_k;
    opts.metric = NULL;
    opts.quantization = usearch_scalar_f32_k;
    return opts;
}

void test_init(size_t collection_size, size_t dimensions) {
    printf("Test: Index Initialization...\n");

    // Init index
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    ASSERT(!error, error);
    usearch_free(idx, &error);
    ASSERT(!error, error);

    // Init second time
    idx = usearch_init(&opts, &error);
    ASSERT(!error, error);

    ASSERT(usearch_size(idx, &error) == 0, error);
    ASSERT(usearch_capacity(idx, &error) == 0, error);
    ASSERT(usearch_dimensions(idx, &error) == dimensions, error);
    ASSERT(usearch_connectivity(idx, &error) == opts.connectivity, error);

    // Reserve
    usearch_reserve(idx, collection_size, &error);
    ASSERT(!error, error);
    ASSERT(usearch_size(idx, &error) == 0, error);
    ASSERT(usearch_capacity(idx, &error) == collection_size, error);
    ASSERT(usearch_dimensions(idx, &error) == dimensions, error);
    ASSERT(usearch_connectivity(idx, &error) == opts.connectivity, error);

    usearch_free(idx, &error);
    ASSERT(!error, error);

    printf("Test: Index Initialization - PASSED\n");
}

void test_add_vector(size_t collection_size, size_t dimensions) {
    printf("Test: Add Vector...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    ASSERT(usearch_size(idx, &error) == collection_size, error);
    ASSERT(usearch_capacity(idx, &error) == collection_size, error);

    // Check vectors in the index
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        ASSERT(usearch_contains(idx, key, &error), error);
    }
    ASSERT(!usearch_contains(idx, -1, &error), error); // Non existing key

    free(data);
    usearch_free(idx, &error);
    printf("Test: Add Vector - PASSED\n");
}

void test_find_vector(size_t collection_size, size_t dimensions) {
    printf("Test: Find Vector...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Create result buffers
    int results_count = collection_size;
    usearch_key_t* keys = (usearch_key_t*)malloc(results_count * sizeof(usearch_key_t));
    float* distances = (float*)malloc(results_count * sizeof(float));
    ASSERT(keys && distances, "Failed to allocate memory");

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Find the vectors
    for (size_t i = 0; i < collection_size; i++) {
        const void* query_vector = data + i * dimensions;
        size_t found_count =
            usearch_search(idx, query_vector, usearch_scalar_f32_k, results_count, keys, distances, &error);
        ASSERT(!error, error);
        ASSERT(found_count = results_count, "Vector is missing");
    }

    free(data);
    free(keys);
    free(distances);
    usearch_free(idx, &error);
    printf("Test: Find Vector - PASSED\n");
}

void test_get_vector(size_t collection_size, size_t dimensions) {
    printf("Test: Get Vector...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    opts.multi = true;
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Create result buffers
    int results_count = collection_size;
    float* vectors = (float*)malloc(results_count * dimensions * sizeof(float));
    ASSERT(vectors, "Failed to allocate memory");

    // Add multiple vectors with SAME key
    usearch_key_t const key = 1;
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; i++) {
        usearch_add(idx, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Retrieve vectors from index
    size_t count = usearch_get(idx, key, collection_size, vectors, usearch_scalar_f32_k, &error);
    ASSERT(count == results_count, "Vector is missing");

    free(vectors);
    free(data);
    usearch_free(idx, &error);

    printf("Test: Get Vector - PASSED\n");
}

void test_remove_vector(size_t collection_size, size_t dimensions) {
    printf("Test: Remove Vector...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Remove the vectors
    for (size_t i = 0; i < collection_size; i++) {
        usearch_key_t key = i;
        usearch_remove(idx, key, &error);
        ASSERT(!error, "Currently, Remove is not supported");
    }

    free(data);
    usearch_free(idx, &error);
    printf("Test: Remove Vector - PASSED\n");
}

void test_save_load(size_t collection_size, size_t dimensions) {
    printf("Test: Save/Load...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Save and free the index
    usearch_save(idx, "usearch_index.bin", &error);
    ASSERT(!error, error);
    usearch_free(idx, &error);
    ASSERT(!error, error);

    // Reinit
    idx = usearch_init(&opts, &error);
    ASSERT(!error, error);
    ASSERT(usearch_size(idx, &error) == 0, error);

    // Load
    usearch_load(idx, "usearch_index.bin", &error);
    ASSERT(!error, error);
    ASSERT(usearch_size(idx, &error) == collection_size, error);
    ASSERT(usearch_capacity(idx, &error) == collection_size, error);
    ASSERT(usearch_dimensions(idx, &error) == dimensions, error);
    ASSERT(usearch_connectivity(idx, &error) == opts.connectivity, error);

    // Check vectors in the index
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        ASSERT(usearch_contains(idx, key, &error), error);
    }

    free(data);
    usearch_free(idx, &error);
    printf("Test: Save/Load - PASSED\n");
}

void test_view(size_t collection_size, size_t dimensions) {
    printf("Test: View...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Save and free the index
    usearch_save(idx, "usearch_index.bin", &error);
    ASSERT(!error, error);
    usearch_free(idx, &error);
    ASSERT(!error, error);

    // Reinit
    idx = usearch_init(&opts, &error);
    ASSERT(!error, error);

    // View
    usearch_view(idx, "usearch_index.bin", &error);
    ASSERT(!error, error);

    free(data);
    usearch_free(idx, &error);
    printf("Test: View - PASSED\n");
}

int main() {

    size_t collection_sizes[] = {11, 512};
    size_t dimensions[] = {83, 1};
    for (size_t idx = 0; idx < sizeof(collection_sizes) / sizeof(collection_sizes[0]); ++idx) {
        for (size_t jdx = 0; jdx < sizeof(dimensions) / sizeof(dimensions[0]); ++jdx) {
            test_init(collection_sizes[idx], dimensions[jdx]);
            test_add_vector(collection_sizes[idx], dimensions[jdx]);
            test_find_vector(collection_sizes[idx], dimensions[jdx]);
            test_get_vector(collection_sizes[idx], dimensions[jdx]);
            test_remove_vector(collection_sizes[idx], dimensions[jdx]);
            test_save_load(collection_sizes[idx], dimensions[jdx]);
            test_view(collection_sizes[idx], dimensions[jdx]);
        }
    }

    return 0;
}
