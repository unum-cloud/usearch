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

/**
 * @brief Creates and initializes vectors with random float values.
 *
 * @param count The number of vectors.
 * @param dimensions The number of dimensions per vector.
 * @return A pointer to the first element of the vectors, that must be @b free-ed afterwards.
 */
float* create_vectors(size_t const count, size_t const dimensions) {
    float* data = (float*)malloc(count * dimensions * sizeof(float));
    ASSERT(data, "Failed to allocate memory");
    for (size_t idx = 0; idx < count * dimensions; ++idx)
        data[idx] = (float)rand() / (float)RAND_MAX;
    return data;
}

usearch_init_options_t create_options(size_t const dimensions) {
    usearch_init_options_t opts;
    opts.connectivity = 3; // 32 in faiss
    opts.dimensions = dimensions;
    opts.expansion_add = 40;    // 40 in faiss
    opts.expansion_search = 16; // 10 in faiss
    opts.metric_kind = usearch_metric_ip_k;
    opts.metric = NULL;
    opts.quantization = usearch_scalar_f32_k;
    return opts;
}

/**
 *  This test is designed to verify the initialization of the index with specific dimensions and ensures that the
 *  associated properties are set correctly. It initializes the index twice, checking for errors at each step, and
 *  performs a reserve operation to pre-allocate space in the index, verifying the correct settings of size, capacity,
 *  dimensions, and connectivity after each operation.
 */
void test_init(size_t const collection_size, size_t const dimensions) {
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

/**
 *  This test validates the addition of vectors to the index. It initializes the index and reserves space for vectors.
 *  It then iteratively adds vectors to the index and checks if the index contains the added vectors by verifying the
 *  size, capacity, and presence of each vector in the index.
 */
void test_add_vector(size_t const collection_size, size_t const dimensions) {
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

/**
 *  This test ensures that vectors added to the index can be correctly found. It initializes the index, reserves space,
 *  and adds vectors. It then performs a search query for each added vector to ensure that the vectors are correctly
 *  found in the index, validating the count of found vectors.
 */
void test_find_vector(size_t const collection_size, size_t const dimensions) {
    printf("Test: Find Vector...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Create result buffers
    usearch_key_t* keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    float* distances = (float*)malloc(collection_size * sizeof(float));
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
        size_t found_count =
            usearch_search(idx, data + i * dimensions, usearch_scalar_f32_k, collection_size, keys, distances, &error);
        ASSERT(!error, error);
        ASSERT(found_count >= 1 && found_count <= collection_size, "Vector is missing");
    }

    free(data);
    free(keys);
    free(distances);
    usearch_free(idx, &error);
    printf("Test: Find Vector - PASSED\n");
}

/**
 *  This test checks the ability of the index to handle multiple vectors associated with the same key. It initializes
 *  the index with the multi-option enabled, reserves space, and adds multiple vectors with the same key. The test then
 *  retrieves vectors associated with the key from the index and checks the count of retrieved vectors.
 */
void test_get_vector(size_t const collection_size, size_t const dimensions) {
    printf("Test: Get Vector...\n");

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    opts.multi = true;
    usearch_index_t idx = usearch_init(&opts, &error);
    usearch_reserve(idx, collection_size, &error);

    // Create result buffers
    float* vectors = (float*)malloc(collection_size * dimensions * sizeof(float));
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
    ASSERT(count == collection_size, "Vector is missing");

    free(vectors);
    free(data);
    usearch_free(idx, &error);

    printf("Test: Get Vector - PASSED\n");
}

/**
 *  This test ensures that vectors can be successfully removed from the index. It initializes the index, reserves space,
 *  and adds vectors. It then iteratively removes each vector from the index and checks for errors. However, note that
 *  the assert in this test expects an error, indicating that the remove functionality is not currently supported.
 */
void test_remove_vector(size_t const collection_size, size_t const dimensions) {
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

/**
 *  This test validates the save and load functionality of the index. It initializes the index, reserves space, and adds
 *  vectors. The index is then saved to a file and freed. A new index is initialized, and the previously saved index is
 *  loaded into it. The test then validates the loaded index properties and ensures that it contains all the vectors
 *  from the saved index.
 */
void test_save_load(size_t const collection_size, size_t const dimensions) {
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

/**
 *  This test is designed to validate the view functionality of the index. It initializes the index, reserves space, and
 *  adds vectors. The index is then saved to a file and freed. A new index is initialized and a view is created from the
 *  saved index file. The test is mainly focused on ensuring that no errors occur during these operations, but it does
 *  not verify the properties or contents of the viewed index.
 */
void test_view(size_t const collection_size, size_t const dimensions) {
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
