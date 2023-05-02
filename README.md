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

### Python

Python bindings are implemented with [`pybind/pybind11`](https://github.com/pybind/pybind11).
Assuming the presence of Global Interpreter Lock in Python, on large insertions we spawn threads in the C++ layer.

```sh
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

## Features

### Bring your Threads

## Performance

## TODO

- JavaScript: Allow calling from "worker threads".
- Rust: Allow passing a custom thread ID.
