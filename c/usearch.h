#ifndef UNUM_USEARCH_H
#define UNUM_USEARCH_H

#include <stdlib.h>

typedef enum {
    usearch_metric_ip_k = 0,
    usearch_metric_l2sq_k = 1,
    usearch_metric_cos_k = 2,
    usearch_metric_haversine_k = 3
} usearch_metric_t;

typedef enum {
    usearch_scalar_f32_k = 0,
    usearch_scalar_f16_k = 1,
    usearch_scalar_f8_k = 2,
} usearch_scalar_kind_t;

void* usearch_new(                                           //
    usearch_metric_t metric, usearch_scalar_kind_t accuracy, //
    int dimensions, int capacity, int connectivity,          //
    int expansion_add, int expansion_search);

void usearch_destroy(void* index);

char const* usearch_save(void* index, char const* path);
char const* usearch_load(void* index, char const* path);
char const* usearch_view(void* index, char const* path);

int usearch_size(void* index);
int usearch_connectivity(void* index);
int usearch_dimensions(void* index);
int usearch_capacity(void* index);

char const* usearch_reserve(void* index, int capacity);
char const* usearch_add(void* index, int label, float const* vec);

char const* usearch_search(        //
    void* index,                   //
    float const* query, int limit, //
    int* labels, float* distances, int* count);

#endif
