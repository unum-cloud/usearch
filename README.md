<h1 align="center">USearch</h1>
<h3 align="center">
C++11 Single Header Vector Search<br/>
Compact, yet Powerful<br/>
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/A6wxt6dS9j"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/linkedin.svg" alt="LinkedIn"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/twitter.svg" alt="Twitter"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/blog.svg" alt="Blog"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://github.com/unum-cloud/usearch"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/github.svg" alt="GitHub"></a>
</p>

- [x] Single C++11 header implementation, easily extendible.
- [x] 4B+ sized space efficient point-clouds with `uint40_t`.
- [x] Half-precision support with [`maratyszcza/fp16`](https://github.com/maratyszcza/fp16).
- [x] View from disk, without loading into RAM.
- [x] Any metric, includes: 
  - [x] Euclidean, Dot-product, Cosine,
  - [x] Jaccard, Hamming, Haversine.
  - [x] Hardware-accelerated [`ashvardanian/simsimd`](https://github.com/ashvardanian/simsimd). 
- [x] Variable dimensionality vectors.
- [x] Don't copy vectors if not needed.
- [x] Bring your threads.
- [x] Multiple vectors per label.
- [x] Python bindings: `pip install usearch`.
- [x] JavaScript bindings: `npm install usearch`.
- [x] Rust bindings: `cargo add usearch`.
- [x] Java bindings: `cloud.unum:usearch` on GitHub.
- [ ] GoLang bindings.
- [ ] Wolfram language bindings.
- [x] For Linux: GCC, Clang.
- [x] For MacOS: Apple Clang.
- [ ] For Windows.
- [ ] Multi-index lookups in Python.
- [ ] Thread-safe `reserve`.
- [ ] Distributed construction.

## Usage

There are two usage patters:

- Bare-bones with `usearch/usearch.hpp`, only available in C++.
- Full-fat version with it's own threads, mutexes, type-punning, quantization, that is available both in C++ and is wrapped for higher-level bindings.

### C++

To use in a C++ project simply copy the `include/usearch/usearch.hpp` header into your project.
Alternatively fetch it with CMake:

```cmake
FetchContent_Declare(usearch GIT_REPOSITORY https://github.com/unum-cloud/usearch.git)
FetchContent_MakeAvailable(usearch)
```

The simple usage example would require including the `unum::usearch` namespace and choosing the right "distance" function.
That can be one of the following templates:

- `cos_gt<float>` for "Cosine" or "Angular" distance.
- `ip_gt<float>` for "Inner Product" or "Dot Product" distance.
- `l2_squared_gt<float>` for the squared "L2" or "Euclidean" distance.
- `jaccard_gt<int>` for "Jaccard" distance between two ordered sets of unique elements.
- `bit_hamming_gt<uint>` for "Hamming" distance, as the number of shared bits in hashes.
- `pearson_correlation_gt<float>` for "Pearson" correlation between probability distributions.
- `haversine_gt<float>` for "Haversine" or "Great Circle" distance between coordinates.

That list is easily extendible, and can include similarity measures for vectors that have a different number of elements/dimensions.
The minimal example would be.

```c++
using namespace unum::usearch;

index_gt<cos_gt<float>> index;
float vec[3] = {0.1, 0.3, 0.2};

index.reserve(10);
index.add(/* label: */ 42, /* vector: */ {&vec, 3});
index.search(
  /* query: */ {&vec, 3}, /* top */ 5 /* results */,
  /* with callback: */ [](std::size_t label, float distance) { });

index.save("index.usearch"); // Serializing to disk
index.load("index.usearch"); // Reconstructing from disk
index.view("index.usearch"); // Memory-mapping from disk
```

The `add` is thread-safe for concurrent index construction.
For advanced users, more compile-time abstraction are available.

```cpp
template <typename metric_at = ip_gt<float>,            //
          typename label_at = std::size_t,              // `uint32_t`, `uuid_t`...
          typename id_at = std::uint32_t,               // `uint40_t`, `uint64_t`...
          typename scalar_at = float,                   // `double`, `half`, `char`...
          typename allocator_at = std::allocator<char>> //
class index_gt;
```

One may also define a custom metric, such as Damerau–Levenshtein distance, to compute the similarity between variable length strings.
The only constraint is the function signature:

```cpp
struct custom_metric_t {
    T operator()(T const* a, T const* b, std::size_t a_length, std::size_t b_length) const;
};
```

### Python

Python bindings are implemented with [`pybind/pybind11`](https://github.com/pybind/pybind11).
Assuming the presence of Global Interpreter Lock in Python, on large insertions we spawn threads in the C++ layer.

```python
$ pip install usearch

import numpy as np
import usearch

index = usearch.Index(
    dim=256, # Define the number of dimensions in input vectors
    metric='cos', # Choose the "metric" or "distance", default = 'ip', optional
    dtype='f16', # Quantize to 'f16' or 'i8q100' if needed, default = 'f32', optional
    connectivity=16, # How frequent should the connections in the graph be, optional
    expansion_add=128, # Control the recall of indexing, optional
    expansion_search=64, # Control the quality of search, optional
)

n = 100
labels = np.array(range(n), dtype=np.longlong)
vectors = np.random.uniform(0, 0.3, (n, index.ndim)).astype(np.float32)

# You can avoid copying the data
# Handy when build 1B+ indexes of memory-mapped files
index.add(labels, vectors, copy=True)
assert len(index) == n

# You can search a batch at once
matches, distances, counts = index.search(vectors, 10)
```

### JavaScript

```js
// npm install usearch

var index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 2 })
assert.equal(index.connectivity(), 16)
assert.equal(index.dimensions(), 2)
assert.equal(index.size(), 0)

index.add(15, new Float32Array([10, 20]))
assert.equal(index.size(), 2)

var results = index.search(new Float32Array([13, 14]), 2)
assert.deepEqual(results.labels, new Uint32Array([15, 16]))
assert.deepEqual(results.distances, new Float32Array([45, 130]))
```

### Rust

Being a systems-programming language, Rust has better control over memory management and concurrency, but lacks function overloading.
Aside from the `add` and `search`, it also provides `add_in_thread` and `search_in_thread` which let users identify the calling thread to use underlying temporary memory more efficiently.

```rust
// cargo add usearch

let quant: &str = "f16";
let index = new_ip(5,  &quant, 0, 0, 0).unwrap();

assert!(index.reserve(10).is_ok());
assert!(index.capacity() >= 10);
assert!(index.connectivity() != 0);
assert_eq!(index.dimensions(), 5);
assert_eq!(index.size(), 0);

let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
let second: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];

assert!(index.add(42, &first).is_ok());
assert!(index.add(43, &second).is_ok());
assert_eq!(index.size(), 2);

// Read back the tags
let results = index.search(&first, 10).unwrap();
assert_eq!(results.count, 2);

// Validate serialization
assert!(index.save("index.usearch").is_ok());
assert!(index.load("index.usearch").is_ok());
assert!(index.view("index.usearch").is_ok());

// There are more "metrics" available
assert!(new_l2(5,  &quant, 0, 0, 0).is_ok());
assert!(new_cos(5,  &quant, 0, 0, 0).is_ok());
assert!(new_haversine(&quant, 0, 0, 0).is_ok());
```

### Java

```java
Index index = new Index.Config().metric("cos").dimensions(2).build();
float vec[] = {10, 20};
index.add(42, vec);
int[] labels = index.search(vec, 5);
```

### GoLang

### Wolfram

## Features

### Bring your Threads

Most AI, HPC, or Big Data packages use some form of a thread pool.
Instead of spawning additional threads within USearch, we focus on thread-safety of the `add` function.

```cpp
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        native.add(label, span_t{vector, dims}, omp_get_thread_num());
```

During initialization we allocate enough temporary memory for all the cores on the machine.
On call, the user can simply supply the identifier of the current thread, making this library easy to integrate with OpenMP and similar tools.

### Go Beyond 4B Entries

### View Larger Indexes from Disk

### Quantize on the Fly

## Performance

Below are the performance numbers for a benchmark running on the 64 cores of AWS `c7g.metal` "Graviton 3"-based instances.
We fix the default configuration in the top line and show the affects of various parameters by changing one parameter at a time.

|  Vectors   | Connectivity | EF @ A | EF @ S | Add, QPS | Search, QPS | Recall @ 1 |
| :--------: | :----------: | :----: | :----: | :------: | :---------: | ---------: |
| `f32` x256 |      16      |  128   |   64   |  75'640  |   131'654   |      99.3% |
|            |              |        |        |          |             |            |
| `f32` x256 |      12      |  128   |   64   |  81'747  |   149'728   |      99.0% |
| `f32` x256 |      32      |  128   |   64   |  64'368  |   104'050   |      99.4% |
|            |              |        |        |          |             |            |
| `f32` x256 |      16      |   64   |   32   | 128'644  |   228'422   |      97.2% |
| `f32` x256 |      16      |  256   |  128   |  39'981  |   69'065    |      99.2% |
|            |              |        |        |          |             |            |
| `f16` x256 |      16      |   64   |   32   | 128'644  |   228'422   |      97.2% |
| `f32` x256 |      16      |  256   |  128   |  39'981  |   69'065    |      99.2% |

All major HNSW implementation share an identical list of hyper-parameters:

- connectivity (often called `M`),
- expansion on additions (often called `efConstruction`),
- expansion on search (often called `ef`).

The default values vary drastically.

|  Library  | Connectivity | EF @ A | EF @ S |
| :-------: | :----------: | :----: | :----: |
| `hnswlib` |      16      |  200   |   10   |
|  `FAISS`  |      32      |   40   |   16   |
| `USearch` |      16      |  128   |   64   |

### Benchmarking

To achieve best results, please compile locally and check out various configuration options.

```sh
cmake -B ./build_release \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_JEMALLOC=1 && \
    make -C ./build_release -j

./build_release/bench --help
```

Which would print the following instructions:

```txt
SYNOPSIS
        ./build_release/bench [--vectors <path>] [--queries <path>] [--neighbors <path>] [-b] [-j
                              <integer>] [-c <integer>] [--expansion-add <integer>]
                              [--expansion-search <integer>] [--native|--f16quant|--i8quant]
                              [--ip|--l2|--cos|--haversine] [-h]

OPTIONS
        --vectors <path>
                    .fbin file path to construct the index

        --queries <path>
                    .fbin file path to query the index

        --neighbors <path>
                    .ibin file path with ground truth

        -b, --big   Will switch to uint40_t for neighbors lists with over 4B entries
        -j, --threads <integer>
                    Uses all available cores by default

        -c, --connectivity <integer>
                    Index granularity

        --expansion-add <integer>
                    Affects indexing depth

        --expansion-search <integer>
                    Affects search depth

        --native    Use raw templates instead of type-punned classes
        --f16quant  Enable `f16_t` quantization
        --i8quant   Enable `int8_t` quantization
        --ip        Choose Inner Product metric
        --l2        Choose L2 Euclidean metric
        --cos       Choose Angular metric
        --haversine Choose Haversine metric
        -h, --help  Print this help information on this tool and exit
```

## TODO

- JavaScript: Allow calling from "worker threads".
- Rust: Allow passing a custom thread ID.
