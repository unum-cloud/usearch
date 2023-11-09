# Contribution Guide

Thank you for coming here!
It's always nice to have third-party contributors 🤗

---

To keep the quality of the code high, we have a set of guidelines common to [all Unum projects](https://github.com/unum-cloud).

- [What's the procedure?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#organizing-software-development)
- [How to organize branches?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#branches)
- [How to style commits?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#commits)

## Before you start

Before building the first time, please pull `git` submodules.
That's how we bring in SimSIMD and other optional dependencies to test all of the available functionality.

```sh
git submodule update --init --recursive
```

## C++ 11 and C 99

Our primary C++ implementation uses CMake for builds.
If this is your first experience with CMake, use the following commands to get started:

```sh
sudo apt-get update && sudo apt-get install cmake build-essential libjemalloc-dev g++-12 gcc-12 # Ubuntu
brew install libomp llvm # MacOS
```

Using modern syntax, this is how you build and run the test suite:

```sh
cmake -DUSEARCH_BUILD_TEST_CPP=1 -B ./build_debug
cmake --build ./build_debug --config Debug
./build_debug/test_cpp
```

The CMakeLists.txt file has a number of options you can pass:

- What to build:
  - `USEARCH_BUILD_TEST_CPP` - build the C++ test suite
  - `USEARCH_BUILD_BENCH_CPP` - build the C++ benchmark suite
  - `USEARCH_BUILD_LIB_C` - build the C library
  - `USEARCH_BUILD_TEST_C` - build the C test suite
- Which dependencies to use:
  - `USEARCH_USE_OPENMP` - use OpenMP for parallelism
  - `USEARCH_USE_SIMSIMD` - use SimSIMD for vectorization
  - `USEARCH_USE_JEMALLOC` - use Jemalloc for memory management
  - `USEARCH_USE_FP16LIB` - use software emulation for half-precision floating point

Putting all of this together on Ubuntu and compiling the "release" version using the GCC 12 compiler:

```sh
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DUSEARCH_USE_FP16LIB=1 \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_SIMSIMD=1 \
    -DUSEARCH_USE_JEMALLOC=1 \
    -DUSEARCH_BUILD_TEST_CPP=1 \
    -DUSEARCH_BUILD_BENCH_CPP=1 \
    -DUSEARCH_BUILD_LIB_C=1 \
    -DUSEARCH_BUILD_TEST_C=1 \
    -B ./build_release

cmake --build ./build_release --config Release
./build_release/test_cpp
./build_release/test_c
```

Similarly, to use the most recent Clang compiler version from HomeBrew on MacOS:

```sh
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DUSEARCH_USE_FP16LIB=1 \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_SIMSIMD=1 \
    -DUSEARCH_USE_JEMALLOC=1 \
    -DUSEARCH_BUILD_TEST_CPP=1 \
    -DUSEARCH_BUILD_BENCH_CPP=1 \
    -DUSEARCH_BUILD_LIB_C=1 \
    -DUSEARCH_BUILD_TEST_C=1 \
    -B ./build_release

cmake --build ./build_release --config Release
./build_release/test_cpp
./build_release/test_c
```

Linting:

```sh
cppcheck --enable=all --force --suppress=cstyleCast --suppress=unusedFunction \
    include/usearch/index.hpp \
    include/index_dense.hpp \
    include/index_plugins.hpp
```

## Python 3

Python bindings are built using PyBind11 and are available on [PyPi](https://pypi.org/project/usearch/).
The compilation settings are controlled by the `setup.py` and are independent from CMake used for C/C++ builds.
Use PyTest to validate the build.

- The `-s` option will disable capturing the logs.
- The `-x` option will exit after first failure to simplify debugging.

```sh
pip install -e . && pytest python/scripts/ -s -x
```

Linting:

```sh
pip install ruff
ruff --format=github --select=E9,F63,F7,F82 --target-version=py37 python
```

Before merging your changes you may want to test your changes against the entire matrix of Python versions USearch supports.
For that you need the `cibuildwheel`, which is tricky to use on MacOS and Windows, as it would target just the local environment.
Still, if you have Docker running on any desktop OS, you can use it to build and test the Python bindings for all Python versions for Linux:

```sh
pip install cibuildwheel
cibuildwheel --platform linux
```

## JavaScript

USearch provides NAPI bindings for NodeJS available on [NPM](https://www.npmjs.com/package/usearch).
The compilation settings are controlled by the `binding.gyp` and are independent from CMake used for C/C++ builds.

```sh
npm install
node --test ./javascript/usearch.test.js
npm publish
```

To compile to WebAssembly make sure you have `emscripten` installed and run the following script:

```sh
emcmake cmake -B ./build -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -s TOTAL_MEMORY=64MB" && emmake make -C ./build
node ./build/usearch.test.js
```

If you don't yet have `emcmake` installed:

```sh
git clone https://github.com/emscripten-core/emsdk.git && ./emsdk/emsdk install latest && ./emsdk/emsdk activate latest && source ./emsdk/emsdk_env.sh
```

## Rust

USearch provides Rust bindings available on [Crates.io](https://crates.io/crates/usearch).
The compilation settings are controlled by the `build.rs` and are independent from CMake used for C/C++ builds.

```sh
cargo test -p usearch
cargo publish
```

## Objective-C and Swift

USearch provides both Objective-C and Swift bindings through the [Swift Package Manager](https://swiftpackageindex.com/unum-cloud/usearch).
The compilation settings are controlled by the `Package.swift` and are independent from CMake used for C/C++ builds.

```sh
swift build
swift test -v
```

## GoLang

USearch provides GoLang bindings, that depend on the C library that must be installed beforehand.
So one should first compile the C library, link it with GoLang, and only then run tests.

```sh
cmake -B ./build_release -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_BUILD_TEST_C=1 -DUSEARCH_USE_OPENMP=1 -DUSEARCH_USE_SIMSIMD=1 
cmake --build ./build_release --config Release -j

mv ./c/libusearch_c.so ./golang/ # or .dylib to install the library on MacOS
cp ./c/usearch.h ./golang/ # to make the header available to GoLang

cd golang && go test -v ; cd ..
```

## Java

USearch provides Java bindings available from the [GitHub Maven registry](https://github.com/unum-cloud/usearch/packages/1867475) and the [Sonatype Maven Central Repository](https://central.sonatype.com/artifact/cloud.unum/usearch).
The compilation settings are controlled by the `build.gradle` and are independent from CMake used for C/C++ builds.

To setup the Gradle environment:

```sh
sudo apt get install zip unzip
curl -s "https://get.sdkman.io" | bash
sdk install java gradle
```

Afterwards, in a new terminal:

```sh
gradle clean build
java -cp . -Djava.library.path="$(pwd)/build/libs/usearch/shared" java/cloud/unum/usearch/Index.java
```

Or step by-step:

```sh
cs java/unum/cloud/usearch
javac -h . Index.java

# Ubuntu:
g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -fPIC -o libusearch_c.so cloud_unum_usearch_Index.o -lc

# Windows
g++ -c -I%JAVA_HOME%\include -I%JAVA_HOME%\include\win32 cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -o USearchJNI.dll cloud_unum_usearch_Index.o -Wl,--add-stdcall-alias

# MacOS
g++ -std=c++11 -c -fPIC \
    -I../../../../include \
    -I../../../../fp16/include \
    -I../../../../simsimd/include \
    -I${JAVA_HOME}/include -I${JAVA_HOME}/include/darwin cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -dynamiclib -o libusearch.dylib cloud_unum_usearch_Index.o -lc
# Run linking to that directory
java -cp . -Djava.library.path="$(pwd)/java/cloud/unum/usearch/" Index.java
java -cp . -Djava.library.path="$(pwd)/java" cloud.unum.usearch.Index
```

## Wolfram

```sh
brew install --cask wolfram-engine
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

## Working on Sub-Modules

Extending metrics in SimSIMD:

```sh
git push --set-upstream https://github.com/ashvardanian/simsimd.git HEAD:main
```
