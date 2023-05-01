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

- [x] Single C++11 header, easily extendible.
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
- [x] CPython bindings for Python 3.
- [ ] Node.js bindings for JavaScript.
- [ ] Wolfram language bindings.
- [x] For Linux: GCC, Clang.
- [x] For MacOS: Apple Clang.
- [ ] For Windows.
- [ ] Distributed construction.
- [ ] Multi-index lookups.

## Usage

### C++

To use in a C++ project simply copy the `include/usearch/usearch.hpp` header into your project.

```c++
using namespace unum::usearch;

index_gt<cos_gt<float>> index;
float vec[3] = {0.1, 0.3, 0.2};
index.add(/* label: */ 42, /* vector: */ {&vec, 3});
index.search(
  /* query: */ {&vec, 3}, 10 /* results */,
  /* callback: */ [](std::size_t label, float distance) { });
```

Alternatively fetch it with CMake:

```cmake
FetchContent_Declare(usearch GIT_REPOSITORY https://github.com/unum-cloud/usearch.git)
FetchContent_MakeAvailable(usearch)
```

## Features

### Bring your Threads

## Performance
