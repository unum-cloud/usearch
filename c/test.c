#include <stdio.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "usearch.h"

#define ASSERT(must_be_true, message)           \
    if (!(must_be_true)) {                      \
        printf("Assert: %s\n", message);        \
        exit(-1);                               \
    }


void parse_vectors_from_file(char const* file_path, size_t* count, size_t* dimension, float** data) {

    // Open the file
    struct stat st = {};
    ASSERT(stat(file_path, &st) == 0, "Failed to get metadata of the file");
    FILE* file = fopen(file_path, "rb");
    ASSERT(file, "Failed to open the file");

    // Read vectors dimension and calculate count
    int dim = 0;
    size_t read = fread(&dim, sizeof(int), 1, file);
    ASSERT(read == 1, "Failed to read from the file");
    ASSERT(st.st_size % ((dim + 1) * sizeof(float)) == 0, "File does not contain a whole number of vectors");
    *dimension = (size_t)dim;
    *count = st.st_size / ((dim + 1) * sizeof(float));

    // Allocate memory for the vectors' data
    *data = (float*)malloc(*count * *dimension * sizeof(float));
    ASSERT(*data, "Failed to allocate memory");

    // Read the data
    for (size_t i = 0; i < *count; ++i) {
        read = fread(*data + i * *dimension, sizeof(float), *dimension, file);
        ASSERT(read == *dimension, "Failed to read from the file");
         if (i == *count - 1)
            break;

        // Skip
        float skip;
        read = fread(&skip, sizeof(float), 1, file);
        ASSERT(read == 1, "Failed to read from the file");
    }

    fclose(file);
}

usearch_init_options_t create_options(size_t vector_dimension) {
    usearch_init_options_t opts;
    opts.connectivity = 2; // 32 in faiss
    opts.dimensions = vector_dimension;
    opts.expansion_add = 40;    // 40 in faiss
    opts.expansion_search = 16; // 10 in faiss
    opts.metric_kind = usearch_metric_ip_k;
    opts.metric = NULL;
    opts.quantization = usearch_scalar_f32_k;
    return opts;
}

void test_init(size_t vectors_count, size_t vector_dimension) {
    printf("Test: Index Initialization...\n");

    // Init index
    usearch_index_t idx = NULL;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(vector_dimension);
    idx = usearch_init(&opts, &error);
    ASSERT(!error, error);
    usearch_free(idx, &error);
    ASSERT(!error, error);

    // Init second time
    idx = usearch_init(&opts, &error);
    ASSERT(!error, error);

    ASSERT(usearch_size(idx, &error) == 0, error);
    ASSERT(usearch_capacity(idx, &error) == 0, error);
    ASSERT(usearch_dimensions(idx, &error) == vector_dimension, error);
    ASSERT(usearch_connectivity(idx, &error) == opts.connectivity, error);

    // Reserve
    usearch_reserve(idx, vectors_count, &error);
    ASSERT(!error, error);
    ASSERT(usearch_size(idx, &error) == 0, error);
    ASSERT(usearch_capacity(idx, &error) == vectors_count, error);
    ASSERT(usearch_dimensions(idx, &error) == vector_dimension, error);
    ASSERT(usearch_connectivity(idx, &error) == opts.connectivity, error);

    usearch_free(idx, &error);
    ASSERT(!error, error);

    printf("Test: Index Initialization - PASSED\n");
}

void test_add_vector(size_t vectors_count, size_t vector_dimension, float const* data) {
    printf("Test: Add Vector...\n");

    usearch_index_t idx = NULL;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(vector_dimension);
    idx = usearch_init(&opts, &error);
    usearch_reserve(idx, vectors_count, &error);

    // Add vectors
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * vector_dimension, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    ASSERT(usearch_size(idx, &error) == vectors_count, error);
    ASSERT(usearch_capacity(idx, &error) == vectors_count, error);

    // Check vectors in the index
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        ASSERT(usearch_contains(idx, key, &error), error);
    }
    ASSERT(!usearch_contains(idx, -1, &error), error); // Non existing key

    usearch_free(idx, &error);
    printf("Test: Add Vector - PASSED\n");
}

void test_find_vector(size_t vectors_count, size_t vector_dimension, float const* data) {
    printf("Test: Find Vector...\n");

    usearch_index_t idx = NULL;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(vector_dimension);
    idx = usearch_init(&opts, &error);
    usearch_reserve(idx, vectors_count, &error);

    // Create result buffers
    int results_count = 10;
    usearch_key_t* keys = (usearch_key_t*)malloc(results_count * sizeof(usearch_key_t));
    float* distances = (float*)malloc(results_count * sizeof(float));
    ASSERT(keys && distances, "Failed to allocate memory");

    // Add vectors
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * vector_dimension, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Find the vectors
    for (size_t i = 0; i < vectors_count; i++) {
        const void *query_vector = data + i * vector_dimension;
        size_t found_count = usearch_search(idx, query_vector, usearch_scalar_f32_k, results_count, keys, distances, &error);
        ASSERT(!error, error);
        ASSERT(found_count = results_count, "Vector is missing");
    }

    usearch_free(idx, &error);
    printf("Test: Find Vector - PASSED\n");
}

void test_remove_vector(size_t vectors_count, size_t vector_dimension, float const* data) {
    printf("Test: Remove Vector...\n");

    usearch_index_t idx = NULL;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(vector_dimension);
    idx = usearch_init(&opts, &error);
    usearch_reserve(idx, vectors_count, &error);

    // Add vectors
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * vector_dimension, usearch_scalar_f32_k, &error);
        ASSERT(!error, error);
    }

    // Remove the vectors
    for (size_t i = 0; i < vectors_count; i++) {
        usearch_key_t key = i;
        usearch_remove(idx, key, &error);
        ASSERT(error, "Currently, Remove is not supported");
    }

    usearch_free(idx, &error);
    printf("Test: Remove Vector - PASSED\n");
}

void test_save_load(size_t vectors_count, size_t vector_dimension, float const* data) {
    printf("Test: Save/Load...\n");

    usearch_index_t idx = NULL;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(vector_dimension);
    idx = usearch_init(&opts, &error);
    usearch_reserve(idx, vectors_count, &error);

    // Add vectors
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * vector_dimension, usearch_scalar_f32_k, &error);
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
    ASSERT(usearch_size(idx, &error) == vectors_count, error);
    ASSERT(usearch_capacity(idx, &error) == vectors_count, error);
    ASSERT(usearch_dimensions(idx, &error) == vector_dimension, error);
    ASSERT(usearch_connectivity(idx, &error) == opts.connectivity, error);

    // Check vectors in the index
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        ASSERT(usearch_contains(idx, key, &error), error);
    }

    usearch_free(idx, &error);
    printf("Test: Save/Load - PASSED\n");
}

void test_view(size_t vectors_count, size_t vector_dimension, float const* data) {
    printf("Test: View...\n");

    usearch_index_t idx = NULL;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(vector_dimension);
    idx = usearch_init(&opts, &error);
    usearch_reserve(idx, vectors_count, &error);

    // Add vectors
    for (size_t i = 0; i < vectors_count; ++i) {
        usearch_key_t key = i;
        usearch_add(idx, key, data + i * vector_dimension, usearch_scalar_f32_k, &error);
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

    usearch_free(idx, &error);
    printf("Test: View - PASSED\n");
}

int main(int argc, char* argv[]) {
    // Resolve file path
    char const* filename = "sample.fvecs";
    if (argc == 2)
        filename = argv[1];

    // Read data from the file
    float* data = NULL;
    size_t vectors_count = 0;
    size_t vector_dimension = 0;
    parse_vectors_from_file(filename, &vectors_count, &vector_dimension, &data);
    vectors_count = vectors_count > 100000 ? 100000 : vectors_count; // Just limit vectors

    // Test
    test_init(vectors_count, vector_dimension);
    test_add_vector(vectors_count, vector_dimension, data);
    test_find_vector(vectors_count, vector_dimension, data);
    test_remove_vector(vectors_count, vector_dimension, data);
    test_save_load(vectors_count, vector_dimension, data);
    test_view(vectors_count, vector_dimension, data);

    // Free vectors' buffer
    free(data);

    return 0;
}