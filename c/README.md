# USearch for C

## Installation

The simplest form to integrate is to copy the contents of `usearch/c/` into your project.
To build the library `libusearch_static_c` and `libusearch_c`, pass enable the `USEARCH_BUILD_CLIB` CMake option:

```bash
cmake -DUSEARCH_BUILD_CLIB=1 -DUSEARCH_BUILD_CTEST=1 -DUSEARCH_BUILD_STATIC=0 -DUSEARCH_BUILD_TEST=0 -DUSEARCH_BUILD_BENCHMARK=0 ..
```

You can choose between a static and dynamic library by setting `USEARCH_BUILD_STATIC`.

## Quickstart

```c
usearch_error_t error = NULL;
usearch_init_options_t opts = create_options(dimensions);
usearch_index_t index = usearch_init(&opts, &error);
usearch_reserve(index, vectors_count, &error);

float vector[dimensions] = {...};
usearch_add(index, 42, &vector[0], usearch_scalar_f32_k, &error);

ASSERT(!error, error);
ASSERT(usearch_size(index, &error) == vectors_count, error);
ASSERT(usearch_capacity(index, &error) == vectors_count, error);

usearch_key_t found_keys[10];
float found_distances[10];
size_t found_count = usearch_search(index, &vector[0], usearch_scalar_f32_k, 10, &found_keys[0], &found_distances[0], &error);

usearch_free(index, &error);
```
