# -----------------------------------------------------------------------------
# Makefile for the USearch C Library
#
# Main Targets:
#   - libusearch_c.a       : Builds the static library.
#   - libusearch_c.so      : Builds the shared library for Linux.
#   - libusearch_c.dylib   : Builds the shared library for macOS.
#   - test_c               : Compiles the test executable.
#   - clean                : Removes generated files.
#
# Macros:
#   - USEARCH_USE_OPENMP   : Enables OpenMP support. Default is 1.
#   - USEARCH_USE_SIMSIMD  : Enables Simd support. Default is 1.
#   - CC                   : The C compiler to use. Default is gcc.
#   - CXX                  : The C++ compiler to use. Default is g++.
#   - C_FLAGS              : Flags for the C compiler.
#   - CXX_FLAGS            : Flags for the C++ compiler.
# -----------------------------------------------------------------------------

CC ?= gcc
CXX ?= g++

# Preset USEARCH_USE_OPENMP and USEARCH_USE_SIMSIMD, if not provided
USEARCH_USE_OPENMP ?= 1
USEARCH_USE_SIMSIMD ?= 1

C_FLAGS = -std=c99
CXX_FLAGS = -std=c++11 -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas -pedantic -O3 -fPIC

CXX_FLAGS += -DUSEARCH_USE_OPENMP=$(USEARCH_USE_OPENMP)
CXX_FLAGS += -DUSEARCH_USE_SIMSIMD=$(USEARCH_USE_SIMSIMD)

# Define source, object files, and libraries
SRCS = lib.cpp
OBJS = $(SRCS:.cpp=.o)
LIBS = -L. -lusearch_c
HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../simsimd/include/

# Default target
.PHONY: all
all: libusearch_c.a libusearch_c.so libusearch_c.dylib test_c

# Builds the static library.
libusearch_c.a: $(OBJS)
	ar rcs $@ $^

# Builds the shared library for Linux.
libusearch_c.so: $(OBJS)
	$(CXX) $(CXX_FLAGS) $(HEADER_INCLUDES) -shared $^ -o $@

# Builds the shared library for macOS.
libusearch_c.dylib: $(OBJS)
	$(CXX) $(CXX_FLAGS) $(HEADER_INCLUDES) -dynamiclib $^ -o $@

# Compiles the test executable.
test_c: test.c libusearch_c.a
	$(CC) $(C_FLAGS) $< $(LIBS) -Wl,-rpath,. -o $@

# Compiles source files to object files.
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(HEADER_INCLUDES) -c $< -o $@

# Removes generated files.
.PHONY: clean
clean:
	rm -f *.o *.a *.so *.dylib test_c

.PHONY: libusearch_c.a libusearch_c.so libusearch_c.dylib test_c
