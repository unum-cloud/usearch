# -----------------------------------------------------------------------------
# Makefile for the USearch C Library
#
# Main Targets:
#   - libusearch_static_c.a        : Builds the static library.
#   - libusearch_c.so              : Builds the shared library for Linux.
#   - libusearch_c.dylib           : Builds the shared library for macOS.
#   - test_c               		   : Compiles the test executable.
#   - clean                		   : Removes generated files.
#
# Macros:
#   - USEARCH_USE_OPENMP   		   : Enables OpenMP support. Default is 1 on Linux, 0 on other OSes.
#   - USEARCH_USE_SIMSIMD  		   : Enables SIMD support. Default is 1.
#   - USEARCH_USE_NATIVE_F16  	   : Enables native half-precision type. Default is 0.
#   - CC                           : The C compiler to use. Default is gcc.
#   - CXX                          : The C++ compiler to use. Default is g++.
#   - C_FLAGS                      : Flags for the C compiler.
#   - CXX_FLAGS                    : Flags for the C++ compiler.
# -----------------------------------------------------------------------------

# Check for available compilers
AVAILABLE_GCC := $(shell command -v g++ 2> /dev/null)
AVAILABLE_CLANG := $(shell command -v clang++ 2> /dev/null)

# If g++ is available, choose it; otherwise, choose clang++ if available; else, default to g++
ifeq ($(AVAILABLE_GCC),)
ifeq ($(AVAILABLE_CLANG),)
CC ?= gcc
CXX ?= g++
else
CC ?= clang
CXX ?= clang++
endif
else
CC ?= gcc
CXX ?= g++
endif

# Depending on the compiler, set the correct C++ library
ifneq (,$(findstring g++,$(CXX)))
    CXX_LIB = -lstdc++
else ifneq (,$(findstring clang++,$(CXX)))
    CXX_LIB = -lc++
endif

C_FLAGS = -std=c99 -fPIC -O3 -ffast-math
CXX_FLAGS = -std=c++11 -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas -pedantic -fPIC -O3 -ffast-math

# Preset USEARCH_USE_OPENMP, USEARCH_USE_SIMSIMD, USEARCH_USE_NATIVE_F16, if not provided
USEARCH_USE_SIMSIMD ?= 1
USEARCH_USE_NATIVE_F16 ?= 0

# Detect the operating system
UNAME_S := $(shell uname -s)

# Enable OpenMP by default only on Linux
ifeq ($(UNAME_S),Linux)
USEARCH_USE_OPENMP ?= 1
else
USEARCH_USE_OPENMP ?= 0
endif

CXX_FLAGS += -DUSEARCH_USE_OPENMP=$(USEARCH_USE_OPENMP)
CXX_FLAGS += -DUSEARCH_USE_SIMSIMD=$(USEARCH_USE_SIMSIMD)
CXX_FLAGS += -DUSEARCH_USE_NATIVE_F16=$(USEARCH_USE_NATIVE_F16)

# Define source, object files, and libraries
SRCS = lib.cpp
OBJS = $(SRCS:.cpp=.o)
USEARCH_HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../simsimd/include/

# Default target
.PHONY: all
all: libusearch_static_c.a libusearch_c.so libusearch_c.dylib libusearch_c.dll test_c

# Builds the static library.
libusearch_static_c.a: $(OBJS)
	ar rcs $@ $^

# Builds the shared library for Linux.
libusearch_c.so: $(OBJS)
	$(CXX) $(CXX_FLAGS) $(USEARCH_HEADER_INCLUDES) -shared $^ $(CXX_LIB) -o $@

# Builds the shared library for macOS.
libusearch_c.dylib: $(OBJS)
	$(CXX) $(CXX_FLAGS) $(USEARCH_HEADER_INCLUDES) -dynamiclib $^ $(CXX_LIB) -o $@

# Builds the shared library for Windows.
libusearch_c.dll: $(OBJS)
	$(CXX) $(CXX_FLAGS) $(USEARCH_HEADER_INCLUDES) -shared $^ $(CXX_LIB) -o $@

# Compiles the test executable.
test_c: test.c libusearch_static_c.a
	$(CC) $(C_FLAGS) $< -L. -lusearch_static_c -lc++ -Wl,-rpath,. -o $@

# Compiles source files to object files.
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(USEARCH_HEADER_INCLUDES) -c $< -o $@

# Removes generated files.
.PHONY: clean
clean:
	rm -f *.o *.a *.so *.dylib test_c

.PHONY: libusearch_static_c.a libusearch_c.so libusearch_c.dylib test_c
