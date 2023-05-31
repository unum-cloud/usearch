<h1 align="center">USearch</h1>
<h3 align="center">
Smaller & Faster Single-File<br/>
Vector Search Engine<br/>
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

<p align="center">
Euclidean • Angular • Jaccard • Hamming • Haversine • User-Defined Metrics
<br/>
<a href="#c++">C++11</a> •
<a href="#python">Python</a> •
<a href="#javascript">JavaScript</a> •
<a href="#java">Java</a> •
<a href="#rust">Rust</a> •
<a href="#objective-c">Objective-C</a> •
<a href="#swift">Swift</a> •
<a href="#golang">GoLang</a> •
<a href="#wolfram">Wolfram</a>
<br/>
Linux • MacOS • Windows
</p>

---

- [x] Industry-leading [performance](#performance).
- [x] Easily-extendible [single C++11 header][usearch-header] implementation.
- [x] [User-defined](#define-custom-metrics) and pre-packaged SIMD-accelerated metrics.
- [x] [Half-precision `f16` and Quarter-precision `f8`](#quantize-on-the-fly) support on any hardware.
- [x] [View from disk](#view-larger-indexes-from-disk), without loading into RAM.
- [x] [4B+](#go-beyond-4b-entries) sized space efficient point-clouds with `uint40_t`.
- [x] Variable dimensionality vectors - for [obscure use-cases][obscure-use-cases].
- [x] [Bring your threads](#bring-your-threads), like OpenMP.
- [x] Multiple vectors per label.
- [ ] Thread-safe `reserve`.
- [x] AI + Vector Search = [Semantic Search](#ai--vector-search--semantic-search).

[usearch-header]: https://github.com/unum-cloud/usearch/blob/main/include/usearch/usearch.hpp
[obscure-use-cases]: https://ashvardanian.com/posts/abusing-vector-search

---

|                    | FAISS                         | USearch                            |
| :----------------- | :---------------------------- | :--------------------------------- |
| Implementation     | 84 K [SLOC][sloc] in `faiss/` | 1 K [SLOC][sloc] in `usearch/`     |
| Supported metrics  | 9 fixed metrics               | Any User-Defined metrics           |
| Supported ID types | `uint32_t`, `uint64_t`        | `uint32_t`, `uint40_t`, `uint64_t` |
| Dependencies       | BLAS, OpenMP                  | None                               |
| Bindings           | SWIG                          | Native                             |
| Acceleration       | Learned Quantization          | Downcasting                        |

FAISS is the industry standard for a high-performance batteries-included vector search engine.
Both USearch and FAISS implement the same HNSW algorithm.
But they differ in a lot of design decisions.
USearch is designed to be compact and broadly compatible without sacrificing performance.

|            | FAISS, `f32` | USearch, `f32` | USearch, `f16` | USearch, `f8` | USearch + Numba, `f32` |
| :--------- | -----------: | -------------: | -------------: | ------------: | ---------------------: |
| Insertions |       76 K/s |         89 K/s |         73 K/s |   **137 K/s** |                 99 K/s |
| Queries    |      118 K/s |        167 K/s |        140 K/s |   **281 K/s** |                165 K/s |
| Recall @1  |          99% |          99.2% |            98% |     **99.2%** |                  99.2% |

> Dataset: 1M vectors sample of the Deep1B dataset.
> Hardware: `c7g.metal` AWS instance with 64 cores and DDR5 memory.
> HNSW was configured with identical hyper-parameters:
> connectivity `M=16`,
> expansion @ construction `efConstruction=128`,
> and expansion @ search `ef=64`.
> Both libraries were compiled for the target architecture.
> Jump to the [Performance Tuning][benchmarking] section to read about the effects of those hyper-parameters.

[sloc]: https://en.wikipedia.org/wiki/Source_lines_of_code
[benchmarking]: https://github.com/unum-cloud/usearch/blob/main/docs/benchmarks.md

## User-Defined Functions

Most vector-search packages focus on just 2 metrics - "Inner Product distance" and "Euclidean distance".
That only partially exhausts the list of possible metrics.
A good example would be the rare [Haversine][haversine] distance, used to compute the distance between geo-spatial coordinates, extending Vector Search into the GIS domain.
Another example would be designing a custom metric for **composite embeddings** concatenated from multiple AI models in real-world applications. 
USearch supports that: [Python](#user-defined-functions-in-python) and [C++](#user-defined-functions-in-c) examples.

![USearch: Vector Search Approaches](https://github.com/unum-cloud/usearch/blob/main/assets/usearch-approaches-white.png?raw=true)

Unlike older approaches indexing high-dimensional spaces, like KD-Trees and Locality Sensitive Hashing, HNSW doesn't require vectors to be identical in length.
They only have to be comparable.
So you can apply it in [obscure][obscure] applications, like searching for similar sets or fuzzy text matching.

[haversine]: https://ashvardanian.com/posts/abusing-vector-search#geo-spatial-indexing
[obscure]: https://ashvardanian.com/posts/abusing-vector-search

## Memory Efficiency, Downcasting, and Quantization

Training a quantization model and dimension-reduction is a common approach to accelerate vector search.
Those, however, are only sometimes reliable, can significantly affect the statistical properties of your data, and require regular adjustments if your distribution shifts.

![USearch uint40_t support](https://github.com/unum-cloud/usearch/blob/main/assets/usearch-neighbor-types.png?raw=true)

Instead, we have focused on high-precision arithmetic over low-precision downcasted vectors.
The same index, and `add` and `search` operations will automatically down-cast or up-cast between `f32_t`, `f16_t`, `f64_t`, and `f8_t` representations, even if the hardware doesn't natively support it.
Continuing the topic of memory-efficiency, we provide a `uint40_t` to allow collection with over 4B+ vectors without allocating 8 bytes for every neighbor reference in the proximity graph.

## View Larger Indexes from Disk

Modern search systems often suggest using different servers to maximize indexing speed and minimize serving costs.
Memory-optimized for the first task, and storage-optimized for the second, if the index can be served from external memory, which USearch can.

|          |    To Build     |        To Serve        |
| :------- | :-------------: | :--------------------: |
| Instance |  u-24tb1.metal  |     is4gen.8xlarge     |
| Price    |    ~ $200/h     |        ~$4.5/h         |
| Memory   | 24 TB RAM + EBS | 192 GB RAM + 30 TB SSD |

There is a 50x difference between the cost of such instances for identical capacity.
Of course, the latency of external memory access will be higher, but it is in part compensated with an excellent prefetching mechanism.

## Usage

There are two usage patters:

1. Bare-bones with `usearch/usearch.hpp`, only available in C++.
2. Full-fat version with it's own threads, mutexes, type-punning, quantization, that is available both in C++ and is wrapped for higher-level bindings.

### C++

#### Installation

To use in a C++ project simply copy the `include/usearch/usearch.hpp` header into your project.
Alternatively fetch it with CMake:

```cmake
FetchContent_Declare(usearch GIT_REPOSITORY https://github.com/unum-cloud/usearch.git)
FetchContent_MakeAvailable(usearch)
```

#### Quickstart

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
    results[i].member.label, results[i].member.vector, results[i].distance;
```

The `add` is thread-safe for concurrent index construction.

#### Serialization

```c++
index.save("index.usearch");
index.load("index.usearch"); // Copying from disk
index.view("index.usearch"); // Memory-mapping from disk
```

#### User-Defined Metrics in C++

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
- `bit_hamming_gt<scalar_t>` for "Hamming" distance, as the number of shared bits in hashes.
- `pearson_correlation_gt<scalar_t>` for "Pearson" correlation between probability distributions.
- `haversine_gt<scalar_t>` for "Haversine" or "Great Circle" distance between coordinates.

#### Multi-Threading

Most AI, HPC, or Big Data packages use some form of a thread pool.
Instead of spawning additional threads within USearch, we focus on the thread safety of `add()` function, simplifying resource management.

```cpp
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        native.add(label, span_t{vector, dims}, add_config_t { .thread = omp_get_thread_num() });
```

During initialization, we allocate enough temporary memory for all the cores on the machine.
On the call, the user can supply the identifier of the current thread, making this library easy to integrate with OpenMP and similar tools.

### Python

#### Installation

```sh
pip install usearch
```

#### Quickstart

```python
import numpy as np
from usearch.index import Index

index = Index(
    ndim=3, # Define the number of dimensions in input vectors
    metric='cos', # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
    dtype='f32', # Quantize to 'f16' or 'f8' if needed, default = 'f32'
    connectivity=16, # How frequent should the connections in the graph be, optional
    expansion_add=128, # Control the recall of indexing, optional
    expansion_search=64, # Control the quality of search, optional
)

vector = np.array([0.2, 0.6, 0.4], dtype=np.float32)
index.add(42, vector)
matches, distances, count = index.search(vector, 10)

assert len(index) == 1
assert count == 1
assert matches[0] == 42
assert distances[0] <= 0.001
```

Python bindings are implemented with [`pybind/pybind11`](https://github.com/pybind/pybind11).
Assuming the presence of Global Interpreter Lock in Python, we spawn threads in the C++ layer on large insertions.

#### Serialization

```py
index.save('index.usearch')
index.load('index.usearch') # Copy the whole index into memory
index.view('index.usearch') # View from disk without loading in memory
```

#### Batch Operations

Adding or querying a batch of entries is identical to adding a single vector.
The difference would be in the shape of the tensors.

```py
n = 100
labels = np.array(range(n), dtype=np.longlong)
vectors = np.random.uniform(0, 0.3, (n, index.ndim)).astype(np.float32)

index.add(labels, vectors, threads=..., copy=...)
matches, distances, counts = index.search(vectors, 10, threads=...)

assert matches.shape[0] == vectors.shape[0]
assert counts[0] <= 10
```

You can also override the default `threads` and `copy` arguments in bulk workloads.
The first controls the number of threads spawned for the task.
The second controls whether the vector itself will be persisted inside the index.
If you can preserve the lifetime of the vector somewhere else, you can avoid the copy.

#### User-Defined Metrics in Python

Assuming the language boundary exists between Python user code and C++ implementation, there are more efficient solutions than passing a Python callable to the engine.
Luckily, with the help of [Numba][numba], we can JIT compile a function with a matching signature and pass it down to the engine.

```py
from numba import cfunc, types, carray

signature = types.float32(
    types.CPointer(types.float32),
    types.CPointer(types.float32),
    types.size_t, types.size_t)

@cfunc(signature)
def python_dot(a, b, n, m):
    a_array = carray(a, n)
    b_array = carray(b, n)
    c = 0.0
    for i in range(n):
        c += a_array[i] * b_array[i]
    return c

index = Index(ndim=ndim, metric=python_dot.address)
```

[numba]: https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks

#### Tooling

```py
from usearch.index import Index
from usearch.io import load_matrix, save_matrix

vectors = load_matrix('deep1B.fbin')
index = Index(ndim=vectors.shape[1])
index.add(labels, vectors)
```

### JavaScript

#### Installation

```sh
npm install usearch
```

#### Quickstart

```js
var index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 3 })
index.add(42, new Float32Array([0.2, 0.6, 0.4]))
var results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10)

assert.equal(index.size(), 1)
assert.deepEqual(results.labels, new Uint32Array([42]))
assert.deepEqual(results.distances, new Float32Array([0]))
```

#### Serialization

```js
index.save('index.usearch')
index.load('index.usearch')
index.view('index.usearch')
```

### Rust

#### Installation

```sh
cargo add usearch
```

#### Quickstart

```rust
let quant: &str = "f16";
let index = new_ip(3, &quant, 0, 0, 0).unwrap();

assert!(index.reserve(10).is_ok());
assert!(index.capacity() >= 10);
assert!(index.connectivity() != 0);
assert_eq!(index.dimensions(), 3);
assert_eq!(index.size(), 0);

let first: [f32; 3] = [0.2, 0.1, 0.2];
let second: [f32; 3] = [0.2, 0.1, 0.2];

assert!(index.add(42, &first).is_ok());
assert!(index.add(43, &second).is_ok());
assert_eq!(index.size(), 2);

// Read back the tags
let results = index.search(&first, 10).unwrap();
assert_eq!(results.count, 2);
```

#### Multi-Threading

```rust
assert!(index.add_in_thread(42, &first, 0).is_ok());
assert!(index.add_in_thread(43, &second, 0).is_ok());
let results = index.search_in_thread(&first, 10, 0).unwrap();
```

Being a systems-programming language, Rust has better control over memory management and concurrency but lacks function overloading.
Aside from the `add` and `search`, USearch Rust binding also provides `add_in_thread` and `search_in_thread`, which let users identify the calling thread to use underlying temporary memory more efficiently.

#### Serialization

```rust
assert!(index.save("index.usearch").is_ok());
assert!(index.load("index.usearch").is_ok());
assert!(index.view("index.usearch").is_ok());
```

#### Metrics

```rust
assert!(new_l2sq(3, &quant, 0, 0, 0).is_ok());
assert!(new_cos(3, &quant, 0, 0, 0).is_ok());
assert!(new_haversine(&quant, 0, 0, 0).is_ok());
```

### Java

#### Installation

```xml
<dependency>
  <groupId>cloud.unum</groupId>
  <artifactId>usearch</artifactId>
  <version>0.2.3</version>
</dependency>
```

Add that snippet to your `pom.xml` and hit `mvn install`.

#### Quickstart

```java
Index index = new Index.Config().metric("cos").dimensions(2).build();
float vec[] = {10, 20};
index.add(42, vec);
int[] labels = index.search(vec, 5);
```

### Swift

#### Installation

```txt
https://github.com/unum-cloud/usearch
```

#### Quickstart

```swift
let index = Index.l2sq(dimensions: 3, connectivity: 8)
let vectorA: [Float32] = [0.3, 0.5, 1.2]
let vectorB: [Float32] = [0.4, 0.2, 1.2]
index.add(label: 42, vector: vectorA[...])
index.add(label: 43, vector: vectorB[...])

let results = index.search(vector: vectorA[...], count: 10)
assert(results.0[0] == 42)
```

### GoLang

#### Installation

```golang
import (
	"github.com/unum-cloud/usearch/golang"
)
```

#### Quickstart

```golang
package main

import (
	"fmt"
	"github.com/unum-cloud/usearch/golang"
)

func main() {
	conf := usearch.DefaultConfig(128)
	index := usearch.NewIndex(conf)
	v := make([]float32, 128)
	index.Add(42, v)
	results := index.Search(v, 1)
}
```

### Wolfram

## TODO

- JavaScript: Allow calling from "worker threads".
- Rust: Allow passing a custom thread ID.
- C# .NET bindings.

## AI + Vector Search = Semantic Search

AI has a growing number of applications, but one of the coolest classic ideas is to use it for Semantic Search.
One can take an encoder model, like the multi-modal UForm, and a web-programming framework, like UCall, and build an image search platform in just 20 lines of Python.

```python
import ucall.rich_posix as ucall
import uform
from usearch.index import Index

import numpy as np
from PIL import Image

server = ucall.Server()
model = uform.get_model('unum-cloud/uform-vl-multilingual')
index = Index(ndim=256)

@server
def add(label: int, photo: Image.Image):
    image = model.preprocess_image(photo)
    vector = model.encode_image(image).detach().numpy()
    index.add(label, vector.flatten(), copy=True)

@server
def search(query: str) -> np.ndarray:
    tokens = model.preprocess_text(query)
    vector = model.encode_text(tokens).detach().numpy()
    neighbors, _, _ = index.search(vector.flatten(), 3)
    return neighbors

server.run()
```

Check [that](https://github.com/ashvardanian/image-search) and [other](https://github.com/unum-cloud/examples) examples on our corporate GitHub 🤗
