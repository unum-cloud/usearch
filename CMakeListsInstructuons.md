# Compiling USearch

## Build

Before building the first time, please pull submodules.

```sh
git submodule update --init --recursive
```

### Linux

```sh
cmake -B ./build_release \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_JEMALLOC=1 && \
    make -C ./build_release -j

# To benchmark:
./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --ip
```

### MacOS

```sh
brew install libomp llvm
cmake \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
     -B ./build_release && \
    make -C ./build_release -j

# To benchmark:
./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --ip     
```

### Python

```sh
pip install -e .
python python/test.py

# To benchmark:
python python/bench.py \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --ip 
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
cs java
javac -h . Index.java

```

### Wolfram

## Benchmarks

### Datasets

### Evaluation

Yandex Text-to-Image embeddings.

```sh
export path_vectors=datasets/t2i_1M/base.1M.fbin
export path_queries=datasets/t2i_1M/query.public.100K.fbin
export path_neighbors=datasets/t2i_1M/groundtruth.public.100K.ibin
./build_release/bench
```

Wikipedia UForm embeddings.

```sh
export path_vectors=datasets/wiki_1M/base.1M.fbin
export path_queries=datasets/wiki_1M/query.public.100K.fbin 
export path_neighbors=datasets/wiki_1M/groundtruth.public.100K.ibin
./build_release/bench
```

### Profiling

With `perf`:

```sh
# Pass environment variables with `-E`, and `-d` for details
sudo -E perf stat -d ./build_release/bench
sudo -E perf mem -d ./build_release/bench
# Sample on-CPU functions for the specified command, at 1 Kilo Hertz:
sudo -E perf record -F 1000 ./build_release/bench
perf record -d -e arm_spe// -- ./build_release/bench
```
