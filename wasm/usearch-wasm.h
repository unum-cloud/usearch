#ifndef __BINDINGS_USEARCH_WASM_H
#define __BINDINGS_USEARCH_WASM_H
#ifdef __cplusplus
extern "C"
{
  #endif
  
  #include <stdint.h>
  #include <stdbool.h>
  
  typedef struct {
    char *ptr;
    size_t len;
  } usearch_wasm_string_t;
  
  void usearch_wasm_string_set(usearch_wasm_string_t *ret, const char *s);
  void usearch_wasm_string_dup(usearch_wasm_string_t *ret, const char *s);
  void usearch_wasm_string_free(usearch_wasm_string_t *ret);
  typedef uint64_t usearch_wasm_size_t;
  typedef uint64_t usearch_wasm_index_t;
  typedef usearch_wasm_string_t usearch_wasm_error_t;
  void usearch_wasm_error_free(usearch_wasm_error_t *ptr);
  typedef uint64_t usearch_wasm_key_t;
  typedef struct {
    usearch_wasm_key_t *ptr;
    size_t len;
  } usearch_wasm_keys_t;
  void usearch_wasm_keys_free(usearch_wasm_keys_t *ptr);
  typedef float usearch_wasm_distance_t;
  typedef struct {
    usearch_wasm_distance_t *ptr;
    size_t len;
  } usearch_wasm_distances_t;
  void usearch_wasm_distances_free(usearch_wasm_distances_t *ptr);
  typedef struct {
    float *ptr;
    size_t len;
  } usearch_wasm_list_float32_t;
  void usearch_wasm_list_float32_free(usearch_wasm_list_float32_t *ptr);
  typedef struct {
    double *ptr;
    size_t len;
  } usearch_wasm_list_float64_t;
  void usearch_wasm_list_float64_free(usearch_wasm_list_float64_t *ptr);
  typedef struct {
    int8_t *ptr;
    size_t len;
  } usearch_wasm_list_s8_t;
  void usearch_wasm_list_s8_free(usearch_wasm_list_s8_t *ptr);
  typedef struct {
    uint8_t *ptr;
    size_t len;
  } usearch_wasm_list_u8_t;
  void usearch_wasm_list_u8_free(usearch_wasm_list_u8_t *ptr);
  typedef struct {
    uint8_t tag;
    union {
      usearch_wasm_list_float32_t floats32;
      usearch_wasm_list_float64_t floats64;
      usearch_wasm_list_s8_t ints8;
      usearch_wasm_list_u8_t bytes;
    } val;
  } usearch_wasm_vector_t;
  #define USEARCH_WASM_VECTOR_FLOATS32 0
  #define USEARCH_WASM_VECTOR_FLOATS64 1
  #define USEARCH_WASM_VECTOR_INTS8 2
  #define USEARCH_WASM_VECTOR_BYTES 3
  void usearch_wasm_vector_free(usearch_wasm_vector_t *ptr);
  typedef uint8_t usearch_wasm_metric_kind_t;
  #define USEARCH_WASM_METRIC_KIND_METRIC_UNKNOWN_K 0
  #define USEARCH_WASM_METRIC_KIND_METRIC_COS_K 1
  #define USEARCH_WASM_METRIC_KIND_METRIC_IP_K 2
  #define USEARCH_WASM_METRIC_KIND_METRIC_L2SQ_K 3
  #define USEARCH_WASM_METRIC_KIND_METRIC_HAVERSINE_K 4
  #define USEARCH_WASM_METRIC_KIND_METRIC_PEARSON_K 5
  #define USEARCH_WASM_METRIC_KIND_METRIC_JACCARD_K 6
  #define USEARCH_WASM_METRIC_KIND_METRIC_HAMMING_K 7
  #define USEARCH_WASM_METRIC_KIND_METRIC_TANIMOTO_K 8
  #define USEARCH_WASM_METRIC_KIND_METRIC_SORENSEN_K 9
  typedef uint8_t usearch_wasm_scalar_kind_t;
  #define USEARCH_WASM_SCALAR_KIND_SCALAR_UNKNOWN_K 0
  #define USEARCH_WASM_SCALAR_KIND_SCALAR_F32_K 1
  #define USEARCH_WASM_SCALAR_KIND_SCALAR_F64_K 2
  #define USEARCH_WASM_SCALAR_KIND_SCALAR_F16_K 3
  #define USEARCH_WASM_SCALAR_KIND_SCALAR_I8_K 4
  #define USEARCH_WASM_SCALAR_KIND_SCALAR_B1_K 5
  typedef struct {
    usearch_wasm_metric_kind_t metric_kind;
    usearch_wasm_scalar_kind_t quantization;
    usearch_wasm_size_t dimensions;
    usearch_wasm_size_t connectivity;
    usearch_wasm_size_t expansion_add;
    usearch_wasm_size_t expansion_search;
  } usearch_wasm_init_options_t;
  usearch_wasm_index_t usearch_wasm_init(usearch_wasm_init_options_t *options, usearch_wasm_error_t *error);
  void usearch_wasm_release(usearch_wasm_index_t index, usearch_wasm_error_t *error);
  void usearch_wasm_save(usearch_wasm_index_t index, usearch_wasm_string_t *path, usearch_wasm_error_t *error);
  void usearch_wasm_load(usearch_wasm_index_t index, usearch_wasm_string_t *path, usearch_wasm_error_t *error);
  void usearch_wasm_view(usearch_wasm_index_t index, usearch_wasm_string_t *path, usearch_wasm_error_t *error);
  usearch_wasm_size_t usearch_wasm_size(usearch_wasm_index_t index, usearch_wasm_error_t *error);
  usearch_wasm_size_t usearch_wasm_capacity(usearch_wasm_index_t index, usearch_wasm_error_t *error);
  usearch_wasm_size_t usearch_wasm_dimensions(usearch_wasm_index_t index, usearch_wasm_error_t *error);
  usearch_wasm_size_t usearch_wasm_connectivity(usearch_wasm_index_t index, usearch_wasm_error_t *error);
  void usearch_wasm_reserve(usearch_wasm_index_t index, usearch_wasm_size_t capacity, usearch_wasm_error_t *error);
  void usearch_wasm_add(usearch_wasm_index_t index, usearch_wasm_key_t key, usearch_wasm_vector_t *array, usearch_wasm_scalar_kind_t vector_kind, usearch_wasm_error_t *error);
  bool usearch_wasm_contains(usearch_wasm_index_t index, usearch_wasm_key_t key, usearch_wasm_error_t *error);
  usearch_wasm_size_t usearch_wasm_search(usearch_wasm_index_t index, usearch_wasm_vector_t *array, usearch_wasm_scalar_kind_t kind, usearch_wasm_size_t results_limit, usearch_wasm_keys_t *found_labels, usearch_wasm_distances_t *found_distances, usearch_wasm_error_t *error);
  bool usearch_wasm_get(usearch_wasm_index_t index, usearch_wasm_key_t key, usearch_wasm_vector_t *array, usearch_wasm_scalar_kind_t vector_kind, usearch_wasm_error_t *error);
  bool usearch_wasm_remove(usearch_wasm_index_t index, usearch_wasm_key_t key, usearch_wasm_error_t *error);
  #ifdef __cplusplus
}
#endif
#endif
