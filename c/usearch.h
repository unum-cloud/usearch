#ifndef UNUM_USEARCH_H
#define UNUM_USEARCH_H

#ifdef __cplusplus
extern "C" {
#endif
#ifndef USEARCH_EXPORT
#define USEARCH_EXPORT
#endif
#include <stdbool.h> // `bool`
#include <stdint.h>  // `size_t`

USEARCH_EXPORT typedef void* usearch_index_t;
USEARCH_EXPORT typedef uint64_t usearch_key_t;
USEARCH_EXPORT typedef float usearch_distance_t;
USEARCH_EXPORT typedef char const* usearch_error_t;

USEARCH_EXPORT typedef usearch_distance_t (*usearch_metric_t)(void const*, void const*);

USEARCH_EXPORT typedef enum usearch_metric_kind_t {
    usearch_metric_unknown_k = 0,
    usearch_metric_ip_k,
    usearch_metric_cos_k,
    usearch_metric_l2sq_k,
    usearch_metric_haversine_k,
    usearch_metric_pearson_k,
    usearch_metric_jaccard_k,
    usearch_metric_hamming_k,
    usearch_metric_tanimoto_k,
    usearch_metric_sorensen_k,
} usearch_metric_kind_t;

USEARCH_EXPORT typedef enum usearch_scalar_kind_t {
    usearch_scalar_unknown_k = 0,
    usearch_scalar_f32_k,
    usearch_scalar_f64_k,
    usearch_scalar_f16_k,
    usearch_scalar_i8_k,
    usearch_scalar_b1_k,
} usearch_scalar_kind_t;

USEARCH_EXPORT typedef struct usearch_init_options_t {
    /**
     *  @brief The metric kind used for distance calculation between vectors.
     */
    usearch_metric_kind_t metric_kind;
    /**
     *  @brief The @b optional custom distance metric function used for distance calculation between vectors.
     *  If the `metric_kind` is set to `usearch_metric_unknown_k`, this function pointer mustn't be `NULL`.
     */
    usearch_metric_t metric;
    /**
     *  @brief The scalar kind used for quantization of vector data during indexing.
     */
    usearch_scalar_kind_t quantization;
    /**
     *  @brief The number of dimensions in the vectors to be indexed.
     *  Must be defined for most metrics, but can be avoided for `usearch_metric_haversine_k`.
     */
    size_t dimensions;
    /**
     *  @brief The @b optional connectivity parameter that limits connections-per-node in graph.
     */
    size_t connectivity;
    /**
     *  @brief The @b optional expansion factor used for index construction when adding vectors.
     */
    size_t expansion_add;
    /**
     *  @brief The @b optional expansion factor used for index construction during search operations.
     */
    size_t expansion_search;
} usearch_init_options_t;

/**
 *  @brief Initializes a new instance of the index.
 *  @param options Pointer to the `usearch_init_options_t` structure containing initialization options.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return A handle to the initialized USearch index, or `NULL` on failure.
 */
USEARCH_EXPORT usearch_index_t usearch_init(usearch_init_options_t* options, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_free(usearch_index_t, usearch_error_t* error);

/**
 *  @brief Saves the index to a file.
 *  @param path The file path where the index will be saved.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_save(usearch_index_t, char const* path, usearch_error_t* error);

/**
 *  @brief Loads the index from a file.
 *  @param path The file path from where the index will be loaded.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_load(usearch_index_t, char const* path, usearch_error_t* error);

/**
 *  @brief Creates a view of the index from a file without loading it into memory.
 *  @param path The file path from where the view will be created.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_view(usearch_index_t, char const* path, usearch_error_t* error);

USEARCH_EXPORT size_t usearch_size(usearch_index_t, usearch_error_t* error);
USEARCH_EXPORT size_t usearch_capacity(usearch_index_t, usearch_error_t* error);
USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t, usearch_error_t* error);
USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t, usearch_error_t* error);

/**
 *  @brief Reserves memory for a specified number of incoming vectors.
 *  @param capacity The desired total capacity including current size.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_reserve(usearch_index_t, size_t capacity, usearch_error_t* error);

/**
 *  @brief Adds a vector with a key to the index.
 *  @param key The key associated with the vector.
 *  @param vector Pointer to the vector data.
 *  @param vector_kind The scalar type used in the vector data.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_add(        //
    usearch_index_t, usearch_key_t key, //
    void const* vector, usearch_scalar_kind_t vector_kind, usearch_error_t* error);

/**
 *  @brief Checks if the index contains a vector with a specific key.
 *  @param key The key to be checked.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return `true` if the index contains the vector with the given key, `false` otherwise.
 */
USEARCH_EXPORT bool usearch_contains(usearch_index_t, usearch_key_t, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search.
 *  @return Number of found matches.
 */
USEARCH_EXPORT size_t usearch_search(                                                                  //
    usearch_index_t, void const* query_vector, usearch_scalar_kind_t query_kind, size_t results_limit, //
    usearch_key_t* found_labels, usearch_distance_t* found_distances, usearch_error_t* error);

/**
 *  @brief Retrieves the vector associated with the given key from the index.
 *  @param key The key of the vector to retrieve.
 *  @param[out] vector Pointer to the memory where the vector data will be copied.
 *  @param vector_kind The scalar type used in the vector data.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return `true` if the vector is successfully retrieved, `false` if the vector is not found.
 */
USEARCH_EXPORT bool usearch_get(        //
    usearch_index_t, usearch_key_t key, //
    void* vector, usearch_scalar_kind_t vector_kind, usearch_error_t* error);

/**
 *  @brief Removes the vector associated with the given key from the index.
 *  @param key The key of the vector to be removed.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return `true` if the vector is successfully removed, `false` if the vector is not found.
 */
USEARCH_EXPORT bool usearch_remove(usearch_index_t, usearch_key_t key, usearch_error_t* error);

#ifdef __cplusplus
}
#endif
#endif
