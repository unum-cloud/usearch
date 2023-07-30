# C-ABI for Usearch

If you depend only on the c interface of usearch, you can include 'usearch/c' into your project directly.
You can also build the c bindings from this folder by running:
```bash
mkdir build
cd build
cmake -DUSEARCH_BUILD_STATIC=ON ..
# cmake -DUSEARCH_BUILD_STATIC=OFF ..
make
```

Alternatively, you can build the shared C library of Usearch (libusearch_c) by running the following in usearch build directory:
```bash
 cmake -DUSEARCH_BUILD_CLIB=1 -DUSEARCH_BUILD_TEST=0 -DUSEARCH_BUILD_BENCHMARK=0 ..
 ```