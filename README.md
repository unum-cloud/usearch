<h1 align="center">USearch</h1>
<h3 align="center">
Faster & Smaller Single-File<br/>
Search Engine for Vectors & Texts<br/>
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
Euclidean • Angular • Bitwise • Haversine • User-Defined Metrics
<br/>
<a href="https://unum-cloud.github.io/usearch/cpp">C++ 11</a> •
<a href="https://unum-cloud.github.io/usearch/python">Python 3</a> •
<a href="https://unum-cloud.github.io/usearch/javascript">JavaScript</a> •
<a href="https://unum-cloud.github.io/usearch/java">Java</a> •
<a href="https://unum-cloud.github.io/usearch/rust">Rust</a> •
<a href="https://unum-cloud.github.io/usearch/c">C 99</a> •
<a href="https://unum-cloud.github.io/usearch/objective-c">Objective-C</a> •
<a href="https://unum-cloud.github.io/usearch/swift">Swift</a> •
<a href="https://unum-cloud.github.io/usearch/csharp">C#</a> •
<a href="https://unum-cloud.github.io/usearch/golang">GoLang</a> •
<a href="https://unum-cloud.github.io/usearch/wolfram">Wolfram</a>
<br/>
Linux • MacOS • Windows • iOS • Docker • WebAssembly
</p>

<div align="center">
<a href="https://pypi.org/project/usearch/"> <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/usearch?label=pypi%20downloads"> </a>
<a href="https://www.npmjs.com/package/usearch"> <img alt="npm" src="https://img.shields.io/npm/dy/usearch?label=npm%20dowloads"> </a>
<a href="https://crates.io/crates/usearch"> <img alt="Crates.io" src="https://img.shields.io/crates/d/usearch?label=crate%20downloads"> </a>
<a href="https://www.nuget.org/packages/Cloud.Unum.USearch"> <img alt="Nuget" src="https://img.shields.io/nuget/dt/Cloud.Unum.USearch?style=social&label=NuGet"> </a>
<a href="https://hub.docker.com/r/unum/usearch"> <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/unum/usearch?style=social&label=Docker"> </a>
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/unum-cloud/usearch">
</div>

---

- ✅ Benchmark-topping performance.
- ✅ Simple and extensible [single C++11 header][usearch-header] implementation.
- ✅ SIMD-optimized and [user-defined metrics](#user-defined-functions) with JIT compilation.
- ✅ Variable dimensionality vectors for unique applications, including search over compressed data.
- ✅ Bitwise Tanimoto and Sorensen coefficients for [Genomics and Chemistry applications](#usearch--rdkit--molecular-search).
- ✅ Hardware-agnostic `f16` & `i8` - [half-precision & quarter-precision support](#memory-efficiency-downcasting-and-quantization).
- ✅ [View large indexes from disk](#serving-index-from-disk) without loading into RAM.
- ✅ Space-efficient point-clouds with `uint40_t`, accommodating 4B+ size.
- ✅ Compatible with OpenMP and custom "executors", for fine-grained control over CPU utilization.
- ✅ Heterogeneous lookups, renaming/relabeling, and on-the-fly deletions.
- ✅ Near-real-time [clustering and sub-clusterings](#clustering) for Tens or Millions of clusters.
- ✅ [Semantic Search](#usearch--ai--multi-modal-semantic-search) and [Joins](#joins).

[usearch-header]: https://github.com/unum-cloud/usearch/blob/main/include/usearch/index.hpp
[obscure-use-cases]: https://ashvardanian.com/posts/abusing-vector-search

## Comparison with FAISS

FAISS is a widely recognized standard for high-performance vector search engines.
USearch and FAISS both employ the same HNSW algorithm, but they differ significantly in their design principles.
USearch is compact and broadly compatible without sacrificing performance, with a primary focus on user-defined metrics and fewer dependencies.

|                     | FAISS                         | USearch                            |
| :------------------ | :---------------------------- | :--------------------------------- |
| Implementation      | 84 K [SLOC][sloc] in `faiss/` | 3 K [SLOC][sloc] in `usearch/`     |
| Supported metrics   | 9 fixed metrics               | Any User-Defined metrics           |
| Supported languages | C++, Python                   | 10 languages                       |
| Supported ID types  | `uint32_t`, `uint64_t`        | `uint32_t`, `uint40_t`, `uint64_t` |
| Dependencies        | BLAS, OpenMP                  | None                               |
| Bindings            | SWIG                          | Native                             |
| Acceleration        | Learned Quantization          | Downcasting                        |

[sloc]: https://en.wikipedia.org/wiki/Source_lines_of_code

Base functionality is identical to FAISS, and the interface must be familiar if you have ever investigated Approximate Nearest Neighbors search:

```py
$ pip install usearch numpy

import numpy as np
from usearch.index import Index

index = Index(
    ndim=3, # Define the number of dimensions in input vectors
    metric='cos', # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
    dtype='f32', # Quantize to 'f16' or 'i8' if needed, default = 'f32'
    connectivity=16, # Optional: How frequent should the connections in the graph be
    expansion_add=128, # Optional: Control the recall of indexing
    expansion_search=64, # Optional: Control the quality of search
)

vector = np.array([0.2, 0.6, 0.4])
index.add(42, vector)
matches: Matches = index.search(vector, 10)

assert len(index) == 1
assert len(matches) == 1
assert matches[0].key == 42
assert matches[0].distance <= 0.001
assert np.allclose(index[42], vector)
```

Comparing the performance of FAISS against USearch on 1 Million 96-dimensional vectors from the famous Deep1B dataset, once can expect the following numbers on modern AWS `c7g.metal` instances.

|              | FAISS, `f32` | USearch, `f32` | USearch, `f16` |     USearch, `i8` |
| :----------- | -----------: | -------------: | -------------: | ----------------: |
| Batch Insert |       16 K/s |         73 K/s |        100 K/s | 104 K/s __+550%__ |
| Batch Search |       82 K/s |        103 K/s |        113 K/s |  134 K/s __+63%__ |
| Bulk Insert  |       76 K/s |        105 K/s |        115 K/s | 202 K/s __+165%__ |
| Bulk Search  |      118 K/s |        174 K/s |        173 K/s | 304 K/s __+157%__ |
| Recall @ 10  |          99% |          99.2% |          99.1% |             99.2% |

> HNSW was configured with identical hyper-parameters:
> connectivity `M=16`,
> expansion @ construction `efConstruction=128`,
> and expansion @ search `ef=64`.
> Batch size is 256.
> Jump to the [Performance Tuning][benchmarking] section to read about the effects of those hyper-parameters.

[benchmarking]: https://github.com/unum-cloud/usearch/blob/main/docs/benchmarks.md

## User-Defined Functions

While most vector search packages concentrate on just a couple of metrics - "Inner Product distance" and "Euclidean distance," USearch extends this list to include any user-defined metrics.
This flexibility allows you to customize your search for a myriad of applications, from computing geo-spatial coordinates with the rare [Haversine][haversine] distance to creating custom metrics for composite embeddings from multiple AI models.

![USearch: Vector Search Approaches](https://github.com/unum-cloud/usearch/blob/main/assets/usearch-approaches-white.png?raw=true)

Unlike older approaches indexing high-dimensional spaces, like KD-Trees and Locality Sensitive Hashing, HNSW doesn't require vectors to be identical in length.
They only have to be comparable.
So you can apply it in [obscure][obscure] applications, like searching for similar sets or fuzzy text matching, using [GZip][gzip-similarity] as a distance function.

> Read more about [JIT and UDF in USearch Python SDK](https://unum-cloud.github.io/usearch/python#user-defined-metrics-and-jit-in-python).

[haversine]: https://ashvardanian.com/posts/abusing-vector-search#geo-spatial-indexing
[obscure]: https://ashvardanian.com/posts/abusing-vector-search
[gzip-similarity]: https://twitter.com/LukeGessler/status/1679211291292889100?s=20

## Memory Efficiency, Downcasting, and Quantization

Training a quantization model and dimension-reduction is a common approach to accelerate vector search.
Those, however, are only sometimes reliable, can significantly affect the statistical properties of your data, and require regular adjustments if your distribution shifts.

![USearch uint40_t support](https://github.com/unum-cloud/usearch/blob/main/assets/usearch-neighbor-types.png?raw=true)

Instead, we have focused on high-precision arithmetic over low-precision downcasted vectors.
The same index, and `add` and `search` operations will automatically down-cast or up-cast between `f32_t`, `f16_t`, `f64_t`, and `i8_t` representations, even if the hardware doesn't natively support it.
Continuing the topic of memory efficiency, we provide a `uint40_t` to allow collection with over 4B+ vectors without allocating 8 bytes for every neighbor reference in the proximity graph.

## Serialization & Serving `Index` from Disk

USearch supports multiple forms of serialization:

- Into a __file__ defined with a path.
- Into a __stream__ defined with a callback, serializing or reconstructing incrementally.
- Into a __buffer__ of fixed length, or a memory-mapped file, that supports random access.

The latter allows you to serve indexes from external memory, enabling you to optimize your server choices for indexing speed and serving costs.
This can result in __20x cost reduction__ on AWS and other public clouds.

```py
index.save("index.usearch")

loaded_copy = index.load("index.usearch")
view = Index.restore("index.usearch", view=True)

other_view = Index(ndim=..., metric=CompiledMetric(...))
other_view.view("index.usearch")
```

## Exact vs. Approximate Search

Approximate search methods, such as HNSW, are predominantly used when an exact brute-force search becomes too resource-intensive.
This typically occurs when you have millions of entries in a collection.
For smaller collections, we offer a more direct approach with the `search` method.

```py
from usearch.index import search, MetricKind, Matches, BatchMatches
import numpy as np

# Generate 10'000 random vectors with 1024 dimensions
vectors = np.random.rand(10_000, 1024).astype(np.float32)
vector = np.random.rand(1024).astype(np.float32)

one_in_many: Matches = search(vectors, vector, 50, MetricKind.L2sq, exact=True)
many_in_many: BatchMatches = search(vectors, vectors, 50, MetricKind.L2sq, exact=True)
```

By passing the `exact=True` argument, the system bypasses indexing altogether and performs a brute-force search through the entire dataset using SIMD-optimized similarity metrics from [SimSIMD](https://github.com/ashvardanian/simsimd).
When compared to FAISS's `IndexFlatL2` in Google Colab, __[USearch may offer up to a 20x performance improvement](https://github.com/unum-cloud/usearch/issues/176#issuecomment-1666650778)__:

- `faiss.IndexFlatL2`: __55.3 ms__.
- `usearch.index.search`: __2.54 ms__.

## `Indexes` for Multi-Index Lookups

For larger workloads targeting billions or even trillions of vectors, parallel multi-index lookups become invaluable.
These lookups prevent the need to construct a single, massive index, allowing users to query multiple smaller ones instead.

```py
from usearch.index import Indexes

multi_index = Indexes(
    indexes: Iterable[usearch.index.Index] = [...],
    paths: Iterable[os.PathLike] = [...],
    view: bool = False,
    threads: int = 0,
)
multi_index.search(...)
```

## Clustering

Once the index is constructed, it can be used to cluster entries much faster.
In essense, the `Index` itself can be seen as a clustering, and it allows iterative deepening.

```py
clustering = index.cluster(
    min_count=10, # Optional
    max_count=15, # Optional
    threads=..., # Optional
)

# Get the clusters and their sizes
centroid_keys, sizes = clustering.centroids_popularity

# Use Matplotlib draw a histogram
clustering.plot_centroids_popularity()

# Export a NetworkX graph of the clusters
g = clustering.network

# Get members of a specific cluster
first_members = clustering.members_of(centroid_keys[0])

# Deepen into that cluster spliting it into more parts, all same arguments supported
sub_clustering = clustering.subcluster(min_count=..., max_count=...)
```

Using Scikit-Learn, on a 1 Million point dataset, one may expect queries to take anywhere from minutes to hours, depending on the number of clusters you want to highlight. For 50'000 clusters the performance difference between USearch and conventional clustering methods may easily reach 100x.

## Joins, One-to-One, One-to-Many, and Many-to-Many Mappings

One of the big questions these days is how will AI change the world of databases and data management.
Most databases are still struggling to implement high-quality fuzzy search, and the only kind of joins they know are deterministic.
A `join` is different from searching for every entry, as it requires a one-to-one mapping, banning collisions among separate search results.

| Exact Search | Fuzzy Search | Semantic Search ? |
| :----------: | :----------: | :---------------: |
|  Exact Join  | Fuzzy Join ? | Semantic Join ??  |

Using USearch one can implement sub-quadratic complexity approximate, fuzzy, and semantic joins.
This can come in handy in any fuzzy-matching tasks, common to Database Management Software.

```py
men = Index(...)
women = Index(...)
pairs: dict = men.join(women, max_proposals=0, exact=False)
```

> Read more in post: [From Dating to Vector Search - "Stable Marriages" on a Planetary Scale 👩‍❤️‍👨](https://ashvardanian.com/posts/searching-stable-marriages)

## Functionality

By now, the core functionality is supported across all bindings.
Broader functionality is ported per request.

|                         | C++ 11 | Python 3 | C 99  | Java  | JavaScript | Rust  | GoLang | Swift |
| :---------------------- | :----: | :------: | :---: | :---: | :--------: | :---: | :----: | :---: |
| Add, search             |   ✅    |    ✅     |   ✅   |   ✅   |     ✅      |   ✅   |   ✅    |   ✅   |
| Save, load, view        |   ✅    |    ✅     |   ✅   |   ✅   |     ✅      |   ✅   |   ✅    |   ✅   |
| User-defined metrics    |   ✅    |    ✅     |   ✅   |   ❌   |     ❌      |   ❌   |   ❌    |   ❌   |
| Joins                   |   ✅    |    ✅     |   ❌   |   ❌   |     ❌      |   ❌   |   ❌    |   ❌   |
| Variable-length vectors |   ✅    |    ❌     |   ❌   |   ❌   |     ❌      |   ❌   |   ❌    |   ❌   |
| 4B+ capacities          |   ✅    |    ❌     |   ❌   |   ❌   |     ❌      |   ❌   |   ❌    |   ❌   |

## Application Examples

### USearch + AI = Multi-Modal Semantic Search

[![USearch Semantic Image Search](https://github.com/ashvardanian/usearch-images/raw/main/assets/usearch-images-slow.gif)](https://github.com/ashvardanian/usearch-images)

AI has a growing number of applications, but one of the coolest classic ideas is to use it for Semantic Search.
One can take an encoder model, like the multi-modal [UForm](https://github.com/unum-cloud/uform), and a web-programming framework, like UCall, and build a text-to-image search platform in just 20 lines of Python.

```python
import ucall
import uform
import usearch

import numpy as np
import PIL as pil

server = ucall.Server()
model = uform.get_model('unum-cloud/uform-vl-multilingual')
index = usearch.index.Index(ndim=256)

@server
def add(key: int, photo: pil.Image.Image):
    image = model.preprocess_image(photo)
    vector = model.encode_image(image).detach().numpy()
    index.add(key, vector.flatten(), copy=True)

@server
def search(query: str) -> np.ndarray:
    tokens = model.preprocess_text(query)
    vector = model.encode_text(tokens).detach().numpy()
    matches = index.search(vector.flatten(), 3)
    return matches.keys

server.run()
```

A more complete [demo with Streamlit is available on GitHub](https://github.com/ashvardanian/usearch-images).
We have pre-processed some commonly used datasets, cleaned the images, produced the vectors, and pre-built the index.

| Dataset                             |            Modalities | Images |                              Download |
| :---------------------------------- | --------------------: | -----: | ------------------------------------: |
| [Unsplash][unsplash-25k-origin]     | Images & Descriptions |   25 K | [HuggingFace / Unum][unsplash-25k-hf] |
| [Conceptual Captions][cc-3m-origin] | Images & Descriptions |    3 M |        [HuggingFace / Unum][cc-3m-hf] |
| [Arxiv][arxiv-2m-origin]            |    Titles & Abstracts |    2 M |     [HuggingFace / Unum][arxiv-2m-hf] |

[unsplash-25k-origin]: https://github.com/unsplash/datasets
[cc-3m-origin]: https://huggingface.co/datasets/conceptual_captions
[arxiv-2m-origin]: https://www.kaggle.com/datasets/Cornell-University/arxiv

[unsplash-25k-hf]: https://huggingface.co/datasets/unum-cloud/ann-unsplash-25k
[cc-3m-hf]: https://huggingface.co/datasets/unum-cloud/ann-cc-3m
[arxiv-2m-hf]: https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m

### USearch + RDKit = Molecular Search

Comparing molecule graphs and searching for similar structures is expensive and slow.
It can be seen as a special case of the NP-Complete Subgraph Isomorphism problem.
Luckily, domain-specific approximate methods exist.
The one commonly used in Chemistry, is to generate structures from [SMILES][smiles], and later hash them into binary fingerprints.
The latter are searchable with bitwise similarity metrics, like the Tanimoto coefficient.
Below is an example using the RDKit package.

```python
from usearch.index import Index, MetricKind
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

molecules = [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO')]
encoder = AllChem.GetRDKitFPGenerator()

fingerprints = np.vstack([encoder.GetFingerprint(x) for x in molecules])
fingerprints = np.packbits(fingerprints, axis=1)

index = Index(ndim=2048, metric=MetricKind.Tanimoto)
keys = np.arange(len(molecules))

index.add(keys, fingerprints)
matches = index.search(fingerprints, 10)
```

[smiles]: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
[rdkit-fingerprints]: https://www.rdkit.org/docs/RDKit_Book.html#additional-information-about-the-fingerprints

### USearch + POI Coordinates = GIS Applications... on iOS?

[![USearch Maps with SwiftUI](https://github.com/ashvardanian/SwiftVectorSearch/raw/main/USearch+SwiftUI.gif)](https://github.com/ashvardanian/SwiftVectorSearch)

With Objective-C and Swift iOS bindings, USearch can be easily used in mobile applications.
The [SwiftVectorSearch](https://github.com/ashvardanian/SwiftVectorSearch) project illustrates how to build a dynamic, real-time search system on iOS.
In this example, we use 2-dimensional vectors—encoded as latitude and longitude—to find the closest Points of Interest (POIs) on a map.
The search is based on the Haversine distance metric, but can easily be extended to support high-dimensional vectors.

## Integrations

- [x] GPTCache: [Python](https://github.com/zilliztech/GPTCache/releases/tag/0.1.29).
- [x] LangChain: [Python](https://github.com/langchain-ai/langchain/releases/tag/v0.0.257) and [JavaScipt](https://github.com/hwchase17/langchainjs/releases/tag/0.0.125).
- [x] ClickHouse: [C++](https://github.com/ClickHouse/ClickHouse/pull/53447).
- [x] Microsoft Semantic Kernel: [Python](https://github.com/microsoft/semantic-kernel/releases/tag/python-0.3.9.dev) and C#.

## Citations

```txt
@software{Vardanian_USearch_2022,
doi = {10.5281/zenodo.7949416},
author = {Vardanian, Ash},
title = {{USearch by Unum Cloud}},
url = {https://github.com/unum-cloud/usearch},
version = {1.0.0},
year = {2022},
month = jun,
}
```
