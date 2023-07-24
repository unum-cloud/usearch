# USearch for C++

## Installation

To use in a C++ project simply copy the `include/usearch/index.hpp` header into your project.
Alternatively fetch it with CMake:

```cmake
FetchContent_Declare(usearch GIT_REPOSITORY https://github.com/unum-cloud/usearch.git)
FetchContent_MakeAvailable(usearch)
```

## Quickstart

Once included, the low-level C++11 interface is as simple as it gets: `reserve()`, `add()`, `search()`, `size()`, `capacity()`, `save()`, `load()`, `view()`.
This covers 90% of use-cases.

```c++
using namespace unum::usearch;

index_gt<cos_gt<float>> index;
float vec[3] = {0.1, 0.3, 0.2};

index.reserve(10);
index.add(/* label: */ 42, /* vector: */ {&vec[0], 3});
auto results = index.search(/* query: */ {&vec[0], 3}, 5 /* neighbors */);

for (std::size_t i = 0; i != results.size(); ++i)
    results[i].element.label, results[i].element.vector, results[i].distance;
```

The `add` is thread-safe for concurrent index construction.

## Serialization

```c++
index.save("index.usearch");
index.load("index.usearch"); // Copying from disk
index.view("index.usearch"); // Memory-mapping from disk
```

## User-Defined Metrics in C++

For advanced users, more compile-time abstractions are available.

```cpp
template <typename metric_at = ip_gt<float>,            //
          typename label_at = std::size_t,              // `uint32_t`, `uuid_t`...
          typename id_at = std::uint32_t,               // `uint40_t`, `uint64_t`...
          typename scalar_at = float,                   // `double`, `half`, `char`...
          typename allocator_at = std::allocator<char>> //
class index_gt;
```

You may want to use a custom memory allocator or a rare scalar type, but most often, you would start by defining a custom similarity measure.
The function object should have the following signature to support different-length vectors.

```cpp
struct custom_metric_t {
    T operator()(T const* a, T const* b, std::size_t a_length, std::size_t b_length) const;
};
```

The following distances are pre-packaged:

- `cos_gt<scalar_t>` for "Cosine" or "Angular" distance.
- `ip_gt<scalar_t>` for "Inner Product" or "Dot Product" distance.
- `l2sq_gt<scalar_t>` for the squared "L2" or "Euclidean" distance.
- `jaccard_gt<scalar_t>` for "Jaccard" distance between two ordered sets of unique elements.
- `hamming_gt<scalar_t>` for "Hamming" distance, as the number of shared bits in hashes.
- `tanimoto_gt<scalar_t>` for "Tanimoto" coefficient for bit-strings.
- `sorensen_gt<scalar_t>` for "Dice-Sorensen" coefficient for bit-strings.
- `pearson_correlation_gt<scalar_t>` for "Pearson" correlation between probability distributions.
- `haversine_gt<scalar_t>` for "Haversine" or "Great Circle" distance between coordinates used in GIS applications.

## Multi-Threading

Most AI, HPC, or Big Data packages use some form of a thread pool.
Instead of spawning additional threads within USearch, we focus on the thread safety of `add()` function, simplifying resource management.

```cpp
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        native.add(label, span_t{vector, dims}, add_config_t { .thread = omp_get_thread_num() });
```

During initialization, we allocate enough temporary memory for all the cores on the machine.
On the call, the user can supply the identifier of the current thread, making this library easy to integrate with OpenMP and similar tools.
