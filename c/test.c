#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "usearch.h"

float* read_fvecs_file(const char* filename, size_t* num_vectors, size_t* vector_dim) {
    struct stat st;
    FILE* file;
    size_t cnt;
    float skip;

    ;
    if (stat(filename, &st) == -1) {
        printf("Failed to read file: %s err: %d: %s\n", filename, errno, strerror(errno));
        return NULL;
    }

    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return NULL;
    }

    // Read the number of vectors and vector dimension from the file
    cnt = fread(vector_dim, sizeof(int), 1, file);
    assert(cnt == 1);
    if (st.st_size % ((*vector_dim + 1) * sizeof(float)) != 0) {
        printf("File does not contain a whole number of vectors.\n Vector dim: %ld, file size: %ld\n", *vector_dim,
               st.st_size);
        fclose(file);
        return NULL;
    }

    *num_vectors = st.st_size / ((*vector_dim + 1) * sizeof(float));

    // Allocate memory for the array
    float* data = (float*)malloc(*num_vectors * *vector_dim * sizeof(float));
    if (data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }

    // Read the data from the file into the array
    for (size_t i = 0; i < *num_vectors; i++) {
        cnt = fread(data + i * *vector_dim, sizeof(float), *vector_dim, file);
        assert(cnt == *vector_dim);
        if (i == *num_vectors - 1) {
            break;
        }
        cnt = fread(&skip, sizeof(float), 1, file);
        assert(cnt == 1);
    }

    fclose(file);
    return data;
}

// log every this many iterations in test loops
const int LOG_GRANULARITY = 10000;

void test_index(float* data, size_t num_vectors, size_t vector_dim) {
    usearch_init_options_t opts;
    usearch_index_t idx;
    usearch_error_t error = NULL;
    clock_t start;
    opts.connectivity = 2; // 32 in faiss
    opts.dimensions = vector_dim;
    opts.expansion_add = 40;    // 40 in faiss
    opts.expansion_search = 16; // 10 in faiss
    opts.metric_kind = usearch_metric_ip_k;
    opts.metric = NULL;
    opts.quantization = usearch_scalar_f32_k;

    // results arrays
    int k = 10;
    num_vectors /= 10;
    usearch_label_t* labels = (usearch_label_t*)calloc(k, sizeof(usearch_label_t));
    float* distances = (float*)calloc(k, sizeof(float));

    fprintf(stderr, "read %ld vectors of dimension %ld\n", num_vectors, vector_dim);
    fprintf(stderr, "initializing usearch...\n");
    idx = usearch_init(&opts, &error);
    if (error != NULL) {
        fprintf(stderr, "usearch error: %s\n", error);
        assert(false);
    }

    fprintf(stderr, "reserving capacity...\n");
    usearch_reserve(idx, num_vectors, &error);
    assert(!error);

    fprintf(stderr, "adding vectors...\n");
    for (size_t i = 0; i < num_vectors; i++) {
        if (i % LOG_GRANULARITY == 0) {
            fprintf(stderr, "added %ld vectors...\n", i);
        }
        usearch_label_t label = i;
        usearch_add(idx, label, data + i * vector_dim, usearch_scalar_f32_k, &error);
        assert(!error);
    }

    fprintf(stderr, "checking containment of vectors...\n");
    for (size_t i = 0; i < num_vectors; i++) {
        if (i % LOG_GRANULARITY == 0) {
            fprintf(stderr, "checked %ld vectors...\n", i);
        }
        usearch_label_t label = i;
        assert(usearch_contains(idx, label, &error));
        assert(!error);
    }

    assert(!usearch_contains(idx, -1, &error));
    assert(!error);

    fprintf(stderr, "searching vectors...\n");
    for (size_t i = 0; i < num_vectors; i++) {
        if (i % LOG_GRANULARITY == 0) {
            fprintf(stderr, "searched %ld vectors...\n", i);
        }
        const void *query_vector = data + i * vector_dim;
        usearch_search(idx, query_vector, usearch_scalar_f32_k, k, labels, distances, &error);
        assert(!error);
    }

    fprintf(stderr, "saving the index...\n");
    start = clock();
    usearch_save(idx, "usearch_index.bin", &error);
    assert(!error);
    fprintf(stderr, "saving took %.2f ms\n", ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000);

    fprintf(stderr, "destroying the index...\n");
    start = clock();
    usearch_free(idx, &error);
    assert(!error);
    fprintf(stderr, "destroying the index took %.2f ms\n", ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000);

    fprintf(stderr, "loading the index...\n");
    start = clock();
    idx = usearch_init(&opts, &error);
    assert(!error);
    assert(usearch_size(idx, &error) == 0);
    assert(!error);
    usearch_load(idx, "usearch_index.bin", &error);
    assert(!error);
    fprintf(stderr, "loading the index took %.2f ms\n", ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000);
    assert(usearch_size(idx, &error) == num_vectors);
    assert(!error);
    usearch_free(idx, &error);

    fprintf(stderr, "viewing the index...\n");
    start = clock();
    idx = usearch_init(&opts, &error);
    assert(!error);
    assert(usearch_size(idx, &error) == 0);
    assert(!error);
    usearch_view(idx, "usearch_index.bin", &error);
    assert(!error);
    fprintf(stderr, "viewing the index took %.2f ms\n", ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000);
    assert(usearch_size(idx, &error) == num_vectors);
    assert(!error);
    usearch_free(idx, &error);

}

int main() {
    const char* filename = "sift/sift_base.fvecs";
    size_t num_vectors = 0, vector_dim = 0;
    float* data = read_fvecs_file(filename, &num_vectors, &vector_dim);
    if (data != NULL) {

        test_index(data, num_vectors, vector_dim);

        free(data); // Remember to free the dynamically allocated memory
    }
    return 0;
}