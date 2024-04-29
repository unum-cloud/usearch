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
cmake -DUSEARCH_BUILD_TEST_CPP=1 -DCMAKE_BUILD_TYPE=Debug -B ./build_debug
cmake --build ./build_debug --config Debug
./build_debug/test_cpp
```

If there build mode is not specified, the default is `Release`.

```sh
cmake -DUSEARCH_BUILD_TEST_CPP=1 -B ./build_release
cmake --build ./build_release --config Release
./build_release/test_cpp
```

The CMakeLists.txt file has a number of options you can pass:

- What to build:
  - `USEARCH_BUILD_TEST_CPP` - build the C++ test suite
  - `USEARCH_BUILD_BENCH_CPP` - build the C++ benchmark suite
  - `USEARCH_BUILD_LIB_C` - build the C library
  - `USEARCH_BUILD_TEST_C` - build the C test suite
  - `USEARCH_BUILD_SQLITE` - build the SQLite extension ([no Windows](https://gist.github.com/zeljic/d8b542788b225b1bcb5fce169ee28c55))
- Which dependencies to use:
  - `USEARCH_USE_OPENMP` - use OpenMP for parallelism
  - `USEARCH_USE_SIMSIMD` - use SimSIMD for vectorization
  - `USEARCH_USE_JEMALLOC` - use JeMalloc for memory management
  - `USEARCH_USE_FP16LIB` - use software emulation for half-precision floating point

Putting all of this together, compiling all targets on most platforms should work with the following snippet:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DUSEARCH_USE_FP16LIB=1 -DUSEARCH_USE_OPENMP=1 -DUSEARCH_USE_SIMSIMD=1 -DUSEARCH_USE_JEMALLOC=1 -DUSEARCH_BUILD_TEST_CPP=1 -DUSEARCH_BUILD_BENCH_CPP=1 -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_BUILD_TEST_C=1 -DUSEARCH_BUILD_SQLITE=0 -B ./build_release

cmake --build ./build_release --config Release
./build_release/test_cpp
./build_release/test_c
```

Similarly, to use the most recent Clang compiler version from HomeBrew on MacOS:

```sh
brew install clang++ clang cmake
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

I'd recommend putting the following breakpoints when debugging the code in GDB:

- `__asan::ReportGenericError` - to detect illegal memory accesses.
- `__ubsan::ScopedReport::~ScopedReport` - to catch undefined behavior.
- `__GI_exit` - to stop at exit points - the end of running any executable.
- `__builtin_unreachable` - to catch all the places where the code is expected to be unreachable.
- `__usearch_raise_runtime_error` - for USearch-specific assertions.

### Cross Compilation

Unlike GCC, LLVM handles cross compilation very easily.
You just need to pass the right `TARGET_ARCH` and `BUILD_ARCH` to CMake.
The [list includes](https://packages.ubuntu.com/search?keywords=crossbuild-essential&searchon=names):

- `crossbuild-essential-amd64` for 64-bit x86
- `crossbuild-essential-arm64` for 64-bit Arm
- `crossbuild-essential-armhf` for 32-bit ARM hard-float
- `crossbuild-essential-armel` for 32-bit ARM soft-float (emulates `float`)
- `crossbuild-essential-riscv64` for RISC-V
- `crossbuild-essential-powerpc` for PowerPC
- `crossbuild-essential-s390x` for IBM Z
- `crossbuild-essential-mips` for MIPS
- `crossbuild-essential-ppc64el` for PowerPC 64-bit little-endian

Here is an example for cross-compiling for Arm64 on an x86_64 machine:

```sh
sudo apt-get update
sudo apt-get install -y clang lld make crossbuild-essential-arm64 crossbuild-essential-armhf
export CC="clang"
export CXX="clang++"
export AR="llvm-ar"
export NM="llvm-nm"
export RANLIB="llvm-ranlib"
export TARGET_ARCH="aarch64-linux-gnu" # Or "x86_64-linux-gnu"
export BUILD_ARCH="arm64" # Or "amd64"

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_TARGET=${TARGET_ARCH} \
    -DCMAKE_CXX_COMPILER_TARGET=${TARGET_ARCH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${BUILD_ARCH} \
    -B build_artifacts
cmake --build build_artifacts --config Release
```

## Python 3

Python bindings are built using PyBind11 and are available on [PyPi](https://pypi.org/project/usearch/).
The compilation settings are controlled by the `setup.py` and are independent from CMake used for C/C++ builds.
To install USearch locally:

```sh
pip install -e .
```

For testing USearch uses PyTest, which is pre-configured in `pyproject.toml`.
Following options are enabled:

- The `-s` option will disable capturing the logs.
- The `-x` option will exit after first failure to simplify debugging.
- The `-p no:warnings` option will suppress and allow warnings.

```sh
pip install pytest pytest-repeat # for repeated fuzzy tests
pytest # if you trust the default settings
pytest python/scripts/ -s -x -p no:warnings # to overwrite the default settings
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
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform macos                   # works only on MacOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root previligies for multi-architecture builds:

```sh
sudo $(which cibuildwheel) --platform linux
```

On Windows and MacOS, to avoid frequent path resolution issues, you may want to use:

```sh
python -m cibuildwheel --platform windows
```

## JavaScript

USearch provides NAPI bindings for NodeJS available on [NPM](https://www.npmjs.com/package/usearch).
The compilation settings are controlled by the `binding.gyp` and are independent from CMake used for C/C++ builds.
If you don't have NPM installed, first the Node Version Manager:

```sh
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
```

Testing and benchmarking:

```sh
npm install -g typescript
npm run build-js
npm test
npm run bench
```

To compile for AWS Lambda you'd need to recompile the binding.
You can test the setup locally, overriding some of the compilation variables in Docker image:

```Dockerfile
FROM public.ecr.aws/lambda/nodejs:18-x86_64
RUN npm init -y
RUN yum install tar git python3 cmake gcc-c++ -y && yum groupinstall "Development Tools" -y

# Assuming AWS Linux 2 uses old compilers:
ENV USEARCH_USE_FP16LIB 1
ENV DUSEARCH_USE_SIMSIMD 1
ENV SIMSIMD_TARGET_HASWELL 1
ENV SIMSIMD_TARGET_SKYLAKE 0
ENV SIMSIMD_TARGET_ICE 0
ENV SIMSIMD_TARGET_SAPPHIRE 0
ENV SIMSIMD_TARGET_NEON 1
ENV SIMSIMD_TARGET_SVE 0

# For specific PR:
# RUN npm install --build-from-source unum-cloud/usearch#pull/302/head
# For specific version:
# RUN npm install --build-from-source usearch@2.8.8
RUN npm install --build-from-source usearch
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
cargo test -p usearch -- --nocapture --test-threads=1
```

Publishing the crate is a bit more complicated than normally.
If you simply pull the repository with submodules and run the following command it will list fewer files than expected:

```sh
cargo package --list --allow-dirty
```

The reason for that is the heuristic that Cargo uses to determine the files to include in the package.

> Regardless of whether exclude or include is specified, the following files are always excluded:
> Any sub-packages will be skipped (any subdirectory that contains a Cargo.toml file).

Assuming both SimSIMD and StringZilla contain their own `Cargo.toml` files, we need to temporarily exclude them from the package.

```sh
mv simsimd/Cargo.toml simsimd/Cargo.toml.bak
mv stringzilla/Cargo.toml stringzilla/Cargo.toml.bak
cargo package --list --allow-dirty
cargo publish

# Revert back
mv simsimd/Cargo.toml.bak simsimd/Cargo.toml
mv stringzilla/Cargo.toml.bak stringzilla/Cargo.toml
```

## Objective-C and Swift

USearch provides both Objective-C and Swift bindings through the [Swift Package Manager](https://swiftpackageindex.com/unum-cloud/usearch).
The compilation settings are controlled by the `Package.swift` and are independent from CMake used for C/C++ builds.

```sh
swift build && swift test -v
```

> Those depend on Apple's `Foundation` library and can only run on Apple devices.

Swift formatting is enforced with `swift-format` default utility from Apple.
To install and run it on all the files in the project, use the following command:

```bash
brew install swift-format
swift-format . -i -r
```

The style is controlled by the `.swift-format` JSON file in the root of the repository.
As there is no standard for Swift formatting, even Apple's own `swift-format` tool and Xcode differ in their formatting rules, and available settings.

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
gradle test
```

Alternatively, to run the `Index.main`:

```sh
java -cp "$(pwd)/build/classes/java/main" -Djava.library.path="$(pwd)/build/libs/usearch/shared" java/cloud/unum/usearch/Index.java
```

Or step by-step:

```sh
cd java/cloud/unum/usearch
javac -h . Index.java NativeUtils.java

# Ensure JAVA_HOME system environment variable has been set
# e.g. export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Ubuntu:
g++ -c -fPIC -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux -I../../../../include cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -shared -fPIC -o libusearch.so cloud_unum_usearch_Index.o -lc

# Windows
g++ -c -I%JAVA_HOME%\include -I%JAVA_HOME%\include\win32 cloud_unum_usearch_Index.cpp -I..\..\..\..\include -o cloud_unum_usearch_Index.o
g++ -shared -o USearchJNI.dll cloud_unum_usearch_Index.o -Wl,--add-stdcall-alias

# MacOS
g++ -std=c++11 -c -fPIC \
    -I../../../../include \
    -I../../../../fp16/include \
    -I../../../../simsimd/include \
    -I${JAVA_HOME}/include -I${JAVA_HOME}/include/darwin cloud_unum_usearch_Index.cpp -o cloud_unum_usearch_Index.o
g++ -dynamiclib -o libusearch.dylib cloud_unum_usearch_Index.o -lc

# Run linking to that directory
cd ../../../..
cp cloud/unum/usearch/libusearch.* .
java -cp . -Djava.library.path="$(pwd)" cloud.unum.usearch.Index
```

## C#

Setup the .NET environment:

```sh
dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org
```

USearch provides CSharp bindings, that depend on the C library that must be installed beforehand.
So one should first compile the C library, link it with CSharp, and only then run tests.

```sh
cmake -B ./build_artifacts -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_BUILD_TEST_C=1 -DUSEARCH_USE_OPENMP=1 -DUSEARCH_USE_SIMSIMD=1 
cmake --build ./build_artifacts --config Release -j
```

Then, on Windows, copy the library to the CSharp project and run the tests:

```sh
mkdir -p ".\csharp\lib\runtimes\win-x64\native"
cp ".\build_artifacts\libusearch_c.dll" ".\csharp\lib\runtimes\win-x64\native"
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
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

## WebAssembly

```sh
export WASI_VERSION=21
export WASI_VERSION_FULL=${WASI_VERSION}.0
wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_VERSION}/wasi-sdk-${WASI_VERSION_FULL}-linux.tar.gz
tar xvf wasi-sdk-${WASI_VERSION_FULL}-linux.tar.gz
```

After the installation, we can pass WASI SDK to CMake as a new toolchain:

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${WASI_SDK_PATH}/share/cmake/wasi-sdk.cmake .
```

## Working on Sub-Modules

Extending metrics in SimSIMD:

```sh
git push --set-upstream https://github.com/ashvardanian/simsimd.git HEAD:main
```
