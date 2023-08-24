#ifndef UNUM_USEARCH_H
#define UNUM_USEARCH_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USEARCH_EXPORT
#if defined(_WIN32) && !defined(__MINGW32__)
#define USEARCH_EXPORT __declspec(dllexport)
#else
#define USEARCH_EXPORT
#endif
#endif

#include <stdbool.h> // `bool`
#include <stdint.h>  // `size_t`

USEARCH_EXPORT typedef void* usearch_index_t;
USEARCH_EXPORT typedef uint64_t usearch_key_t;
USEARCH_EXPORT typedef float usearch_distance_t;

/**
 *  @brief  Pointer to a null-terminated error message.
 *          Returned error messages @b don't need to be deallocated.
 */
USEARCH_EXPORT typedef char const* usearch_error_t;

/**
 *  @brief  Type-punned callback for "metrics" or "distance functions",
 *          that accepts pointers to two vectors and measures their @b dis-similarity.
 */
USEARCH_EXPORT typedef usearch_distance_t (*usearch_metric_t)(void const*, void const*);

/**
 *  @brief  Enumerator for the most common kinds of `usearch_metric_t`.
 *          Those are supported out of the box, with SIMD-optimizations for most common hardware.
 */
USEARCH_EXPORT typedef enum usearch_metric_kind_t {
    usearch_metric_unknown_k = 0,
    usearch_metric_cos_k,
    usearch_metric_ip_k,
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
    /**
     *  @brief When set allows multiple vectors to map to the same key.
     */
    bool multi;
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
 *  @param[in] path The file path where the index will be saved.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_save(usearch_index_t, char const* path, usearch_error_t* error);

/**
 *  @brief Loads the index from a file.
 *  @param[in] path The file path from where the index will be loaded.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_load(usearch_index_t, char const* path, usearch_error_t* error);

/**
 *  @brief Creates a view of the index from a file without loading it into memory.
 *  @param[in] path The file path from where the view will be created.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_view(usearch_index_t, char const* path, usearch_error_t* error);

USEARCH_EXPORT size_t usearch_size(usearch_index_t, usearch_error_t* error);
USEARCH_EXPORT size_t usearch_capacity(usearch_index_t, usearch_error_t* error);
USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t, usearch_error_t* error);
USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t, usearch_error_t* error);

/**
 *  @brief Reserves memory for a specified number of incoming vectors.
 *  @param[in] capacity The desired total capacity including current size.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_reserve(usearch_index_t, size_t capacity, usearch_error_t* error);

/**
 *  @brief Adds a vector with a key to the index.
 *  @param[in] key The key associated with the vector.
 *  @param[in] vector Pointer to the vector data.
 *  @param[in] vector_kind The scalar type used in the vector data.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_add(        //
    usearch_index_t, usearch_key_t key, //
    void const* vector, usearch_scalar_kind_t vector_kind, usearch_error_t* error);

/**
 *  @brief Checks if the index contains a vector with a specific key.
 *  @param[in] key The key to be checked.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return `true` if the index contains the vector with the given key, `false` otherwise.
 */
USEARCH_EXPORT bool usearch_contains(usearch_index_t, usearch_key_t, usearch_error_t* error);

/**
 *  @brief Counts the number of entries in the index under a specific key.
 *  @param[in] key The key to be checked.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that key.
 */
USEARCH_EXPORT size_t usearch_count(usearch_index_t, usearch_key_t, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to query.
 *  @param[in] query_vector Pointer to the query vector data.
 *  @param[in] query_kind The scalar type used in the query vector data.
 *  @param[in] count Upper bound on the number of neighbors to search, the "k" in "kANN".
 *  @param[out] keys Output buffer for up to `count` nearest neighbors keys.
 *  @param[out] distances Output buffer for up to `count` distances to nearest neighbors.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of found matches.
 */
USEARCH_EXPORT size_t usearch_search(                           //
    usearch_index_t,                                            //
    void const* query_vector, usearch_scalar_kind_t query_kind, //
    size_t count, usearch_key_t* keys, usearch_distance_t* distances, usearch_error_t* error);

/**
 *  @brief Retrieves the vector associated with the given key from the index.
 *  @param[in] key The key of the vector to retrieve.
 *  @param[out] vector Pointer to the memory where the vector data will be copied.
 *  @param[in] count Number of vectors that can be fitted into `vector` for multi-vector entries.
 *  @param[in] vector_kind The scalar type used in the vector data.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that name and exported to `vector`.
 */
USEARCH_EXPORT size_t usearch_get(                    //
    usearch_index_t, usearch_key_t key, size_t count, //
    void* vector, usearch_scalar_kind_t vector_kind, usearch_error_t* error);

/**
 *  @brief Removes the vector associated with the given key from the index.
 *  @param[in] key The key of the vector to be removed.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that name and dropped from the index.
 */
USEARCH_EXPORT size_t usearch_remove(usearch_index_t, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Renames the vector to map to a different key.
 *  @param[in] from The key of the vector to be renamed.
 *  @param[in] to New key for found entry.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that name and renamed.
 */
USEARCH_EXPORT size_t usearch_rename(usearch_index_t, usearch_key_t from, usearch_key_t to, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_H
