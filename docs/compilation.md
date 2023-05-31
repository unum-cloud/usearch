# Compiling USearch

Before building the first time, please pull submodules.

```sh
git submodule update --init --recursive
```

## C++

Linux:

```sh
cmake -B ./build_release \
    -DCMAKE_CXX_COMPILER="g++-12" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_USE_JEMALLOC=1 \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_SIMSIMD=1 \
    -DUSEARCH_BUILD_BENCHMARK=1 \
    -DUSEARCH_BUILD_TEST=1 \
    && \
    make -C ./build_release -j
```

MacOS:

```sh
brew install libomp llvm
cmake -B ./build_release \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_SIMSIMD=1 \
    -DUSEARCH_BUILD_BENCHMARK=1 \
    -DUSEARCH_BUILD_TEST=1 \
    && \
    make -C ./build_release -j
```

Linting:

```sh
cppcheck --enable=all --suppress=cstyleCast --suppress=unusedFunction include/usearch/usearch.hpp src/punned.hpp
```

Benchmarking:

```sh
./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin

./build_release/bench \
    --vectors datasets/t2i_1B/base.1B.fbin \
    --queries datasets/t2i_1B/query.public.100K.fbin \
    --neighbors datasets/t2i_1B/groundtruth.public.100K.ibin \
    --output datasets/t2i_1B/index.usearch \
    --cos
```

## Python

```sh
pip install -e . && pytest python/scripts/test.py
```

Linting:

```sh
pip install ruff
ruff --format=github --select=E9,F63,F7,F82 --target-version=py37 python
```

Benchmarking:

```sh
pip install faiss-cpu
python python/scripts/bench.py \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin
```

> Optional parameters include `connectivity`, `expansion_add`, `expansion_search`.

Checking the effect of different embedding dimensions on construction speed:

```sh
python python/scripts/bench_params.py dimensions
python python/scripts/bench_params.py connectivity
```

## JavaScript

```sh
npm install
node javascript/test.js
npm publish
```

## Rust

```sh
cargo test -p usearch
cargo publish
```

## Java

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
    -I../../../../include \
    -I../../../../src \
    -I../../../../fp16/include \
    -I../../../../robin-map/include \
    -I../../../../simsimd/include \
    -I${JAVA_HOME}/include -I${JAVA_HOME}/include/darwin cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -dynamiclib -o libusearch.dylib cloud_unum_usearch_Index.o -lc
# Run linking to that directory
java -cp . -Djava.library.path=/Users/av/github/usearch/java/cloud/unum/usearch/ Index.java
java -cp . -Djava.library.path=/Users/av/github/usearch/java cloud.unum.usearch.Index
```

## Wolfram

```sh
brew install --cask wolfram-engine
```

## C

# Linux
```sh
g++ -shared -fPIC lib.cpp -I ../include/  -I ../fp16/include/ -o libusearch.so
```
## Docker

```sh
docker build -t unum/usearch . && docker run unum/usearch
```

For multi-architecture builds and publications:

```sh
version=$(cat VERSION)
docker buildx create --use &&
    docker login &&
    docker buildx build \
        --platform "linux/amd64,linux/arm64" \
        --build-arg version=$version \
        --file Dockerfile \
        --tag unum/usearch:$version \
        --tag unum/usearch:latest \
        --push .
```