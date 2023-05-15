# Compiling USearch

## Build

Before building the first time, please pull submodules.

```sh
git submodule update --init --recursive
```

### Linux

```sh
cmake -B ./build_release \
    -DCMAKE_CXX_COMPILER="g++-12" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_JEMALLOC=1 && \
    make -C ./build_release -j

# To benchmark in C++:
./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin

# Large benchmark on a 1TB dataset:
./build_release/bench \
    --vectors datasets/t2i_1B/base.1B.fbin \
    --queries datasets/t2i_1B/query.public.100K.fbin \
    --neighbors datasets/t2i_1B/groundtruth.public.100K.ibin \
    --output datasets/t2i_1B/index.usearch \
    --cos

# To benchmark in Python:
python python/bench.py \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin
```

### MacOS

```sh
brew install libomp llvm
cmake \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DUSEARCH_USE_SIMD=1 \
    -DUSEARCH_USE_OPENMP=1 \
     -B ./build_release && \
    make -C ./build_release -j 
```

### Python

```sh
pip install -e .
python python/test.py
```

### JavaScript

```sh
npm install
node javascript/test.js
npm publish
```

### Rust

```sh
cargo test -p usearch
cargo publish
```

### Java

```sh
gradle clean build
java -cp . -Djava.library.path=/Users/av/github/usearch/build/libs/usearch/shared java/cloud/unum/usearch/Index.java
```

Or step by-step:

```sh
cs java/unum/cloud/usearch
javac -h . Index.java

# Ubuntu:
g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -fPIC -o libusearch.so cloud_unum_usearch_Index.o -lc

# Windows
g++ -c -I%JAVA_HOME%\include -I%JAVA_HOME%\include\win32 cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -o USearchJNI.dll cloud_unum_usearch_Index.o -Wl,--add-stdcall-alias

# MacOS
g++ -std=c++11 -c -fPIC \
    -I../../../../include -I../../../../src -I../../../../fp16/include -I../../../../simsimd/include \
    -I${JAVA_HOME}/include -I${JAVA_HOME}/include/darwin cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -dynamiclib -o libusearch.dylib cloud_unum_usearch_Index.o -lc
# Run linking to that directory
java -cp . -Djava.library.path=/Users/av/github/usearch/java/cloud/unum/usearch/ Index.java
java -cp . -Djava.library.path=/Users/av/github/usearch/java cloud.unum.usearch.Index
```

### Wolfram

```sh
brew install --cask wolfram-engine
```

## Benchmarks

### Datasets

BigANN benchmark is a good starting point, if you are searching for large collections of high-dimensional vectors.
Those often come with precomputed ground-truth neighbors, which is handy for recall evaluation.

| Dataset                                 | Scalar Type | Dimensions | Metric |   Size    |
| :-------------------------------------- | :---------: | :--------: | :----: | :-------: |
| [Unum UForm Wiki][unum-wiki]            |   float32   |    256     |   IP   |   1 GB    |
| [Yandex Text-to-Image Sample][unum-t2i] |   float32   |    200     |  Cos   |   1 GB    |
|                                         |             |            |        |           |
| [Microsoft SPACEV][spacev]              |    int8     |    100     |   L2   |   93 GB   |
| [Microsoft Turing-ANNS][turing]         |   float32   |    100     |   L2   |  373 GB   |
| [Yandex Deep1B][deep]                   |   float32   |     96     |   L2   |  358 GB   |
| [Yandex Text-to-Image][t2i]             |   float32   |    200     |  Cos   |  750 GB   |
|                                         |             |            |        |           |
| [ViT-L/12 LAION][laion]                 |   float32   |    2048    |  Cos   | 2 - 10 TB |

Luckily, smaller samples of those datasets are available.

[unum-wiki]: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
[unum-t2i]: https://huggingface.co/datasets/unum-cloud/ann-t2i-1m
[spacev]: https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B
[turing]: https://learning2hash.github.io/publications/microsoftturinganns1B/
[t2i]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[deep]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[laion]: https://laion.ai/blog/laion-5b/#download-the-data

### Unum UForm Wiki

```sh
mkdir -p datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

### Yandex Text-to-Image

```sh
mkdir -p datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i_1B/
```

### Yandex Deep1B

```sh
mkdir -p datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_1B/
```

### Profiling

With `perf`:

```sh
# Pass environment variables with `-E`, and `-d` for details
sudo -E perf stat -d ./build_release/bench ...
sudo -E perf mem -d ./build_release/bench ...
# Sample on-CPU functions for the specified command, at 1 Kilo Hertz:
sudo -E perf record -F 1000 ./build_release/bench ...
perf record -d -e arm_spe// -- ./build_release/bench ..
```

