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
cs java/unum/cloud/usearch
javac -h . Index.java

# Ubuntu:
g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -fPIC -o libUSearchJNI.so cloud_unum_usearch_Index.o -lc

# Windows
g++ -c -I%JAVA_HOME%\include -I%JAVA_HOME%\include\win32 cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -o USearchJNI.dll cloud_unum_usearch_Index.o -Wl,--add-stdcall-alias

# MacOS
g++ -std=c++11 -c -fPIC \
    -I../../../../include -I../../../../src -I../../../../fp16/include -I../../../../simsimd/include \
    -I${JAVA_HOME}/include -I${JAVA_HOME}/include/darwin cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -dynamiclib -o libUSearchJNI.dylib cloud_unum_usearch_Index.o -lc
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
