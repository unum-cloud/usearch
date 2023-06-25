#ifndef UNUM_USEARCH_H
#define UNUM_USEARCH_H
// in case the header is included from cpp code
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h> // `bool`
#include <stdint.h>  // `size_t`

typedef void* usearch_index_t;
typedef uint32_t usearch_label_t;
typedef float usearch_distance_t;
typedef char const* usearch_error_t;

typedef usearch_distance_t (*usearch_metric_t)(void const*, void const*);

typedef enum usearch_metric_kind_t {
    usearch_metric_ip_k = 0,
    usearch_metric_l2sq_k,
    usearch_metric_cos_k,
    usearch_metric_haversine_k,
    usearch_metric_pearson_k,
    usearch_metric_jaccard_k,
    usearch_metric_hamming_k,
    usearch_metric_tanimoto_k,
    usearch_metric_sorensen_k,
    usearch_metric_unknown_k,
} usearch_metric_kind_t;

typedef enum usearch_scalar_kind_t {
    usearch_scalar_f32_k = 0,
    usearch_scalar_f64_k,
    usearch_scalar_f16_k,
    usearch_scalar_f8_k,
    usearch_scalar_b1_k,
    usearch_scalar_unknown_k,
} usearch_scalar_kind_t;

typedef struct usearch_init_options_t {

    usearch_metric_kind_t metric_kind;
    usearch_metric_t metric;

    usearch_scalar_kind_t quantization;
    size_t dimensions;
    size_t connectivity;
    size_t expansion_add;
    size_t expansion_search;
} usearch_init_options_t;

usearch_index_t usearch_init(usearch_init_options_t*, usearch_error_t*);
void usearch_free(usearch_index_t, usearch_error_t*);

void usearch_save(usearch_index_t, char const* path, usearch_error_t*);
void usearch_load(usearch_index_t, char const* path, usearch_error_t*);
void usearch_view(usearch_index_t, char const* path, usearch_error_t*);

size_t usearch_size(usearch_index_t, usearch_error_t*);
size_t usearch_capacity(usearch_index_t, usearch_error_t*);
size_t usearch_dimensions(usearch_index_t, usearch_error_t*);
size_t usearch_connectivity(usearch_index_t, usearch_error_t*);

void usearch_reserve(usearch_index_t, size_t capacity, usearch_error_t*);

void usearch_add(                                                                            //
    usearch_index_t, usearch_label_t, void const* vector, usearch_scalar_kind_t vector_kind, //
    usearch_error_t*);

bool usearch_contains(usearch_index_t, usearch_label_t, usearch_error_t*);

/**
 *  @brief      Performs k-Approximate Nearest Neighbors Search.
 *  @return     Number of found matches.
 */
size_t usearch_search(                                                                                 //
    usearch_index_t, void const* query_vector, usearch_scalar_kind_t query_kind, size_t results_limit, //
    usearch_label_t* found_labels, usearch_distance_t* found_distances, usearch_error_t*);

bool usearch_get(                     //
    usearch_index_t, usearch_label_t, //
    void* vector, usearch_scalar_kind_t vector_kind, usearch_error_t*);

void usearch_remove(usearch_index_t, usearch_label_t, usearch_error_t*);

#ifdef __cplusplus
}
#endif
#endif
