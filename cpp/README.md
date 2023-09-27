# USearch for C++

## Installation

To use in a C++ project, copy the `include/usearch/*` headers into your project.
Alternatively, fetch it with CMake:

```cmake
FetchContent_Declare(usearch GIT_REPOSITORY https://github.com/unum-cloud/usearch.git)
FetchContent_MakeAvailable(usearch)
```

## Quickstart

Once included, the high-level C++11 interface is as simple as it gets: `reserve()`, `add()`, `search()`, `size()`, `capacity()`, `save()`, `load()`, `view()`.
This covers 90% of use cases.

```cpp
using namespace unum::usearch;

metric_punned_t metric(256, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);

// If you plan to store more than 4 Billion entries - use `index_dense_big_t`.
// Or directly instantiate the template variant you need - `index_dense_gt<key_t, internal_id_t>`.
index_dense_t index = index_dense_t::make(metric);
float vec[3] = {0.1, 0.3, 0.2};

index.reserve(10);
index.add(/* key: */ 42, /* vector: */ {&vec[0], 3});
auto results = index.search(/* query: */ {&vec[0], 3}, 5 /* neighbors */);

for (std::size_t i = 0; i != results.size(); ++i)
    results[i].element.key, results[i].element.vector, results[i].distance;
```

The `add` is thread-safe for concurrent index construction.
It also has an overload for different vector types, casting them under the hood.
The same applies to the `search`, `get`, `cluster`, and `distance_between` functions.

```cpp
double vec_double[3] = {0.1, 0.3, 0.2};
_Float16 vec_half[3] = {0.1, 0.3, 0.2};
index.add(43, {&vec_double[0], 3});
index.add(44, {&vec_half[0], 3});
```

## Serialization

```cpp
index.save("index.usearch");
index.load("index.usearch"); // Copying from disk
index.view("index.usearch"); // Memory-mapping from disk
```

## User-Defined Metrics in C++

For advanced users, more compile-time abstractions are available.

```cpp
template <typename distance_at = default_distance_t,              // `float`
          typename key_at = default_key_t,                        // `int64_t`, `uuid_t`
          typename compressed_slot_at = default_slot_t,           // `uint32_t`, `uint40_t`
          typename dynamic_allocator_at = std::allocator<byte_t>, //
          typename tape_allocator_at = dynamic_allocator_at>      //
class index_gt;
```

The following distances are pre-packaged:

- `metric_cos_gt<scalar_t>` for "Cosine" or "Angular" distance.
- `metric_ip_gt<scalar_t>` for "Inner Product" or "Dot Product" distance.
- `metric_l2sq_gt<scalar_t>` for the squared "L2" or "Euclidean" distance.
- `metric_jaccard_gt<scalar_t>` for "Jaccard" distance between two ordered sets of unique elements.
- `metric_hamming_gt<scalar_t>` for "Hamming" distance, as the number of shared bits in hashes.
- `metric_tanimoto_gt<scalar_t>` for "Tanimoto" coefficient for bit-strings.
- `metric_sorensen_gt<scalar_t>` for "Dice-Sorensen" coefficient for bit-strings.
- `metric_pearson_gt<scalar_t>` for "Pearson" correlation between probability distributions.
- `metric_haversine_gt<scalar_t>` for "Haversine" or "Great Circle" distance between coordinates used in GIS applications.

## Multi-Threading

Most AI, HPC, or Big Data packages use some form of a thread pool.
Instead of spawning additional threads within USearch, we focus on the thread safety of `add()` function, simplifying resource management.

```cpp
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        native.add(key, span_t{vector, dims});
```

During initialization, we allocate enough temporary memory for all the cores on the machine.
On the call, the user can supply the identifier of the current thread, making this library easy to integrate with OpenMP and similar tools.
