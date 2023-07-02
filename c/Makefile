CXX = g++
CC = gcc

C_CXX_FLAGS = -pedantic -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas
C_CXX_FLAGS += -O3 -march=native

C_FLAGS = -std=c99

CXX_FLAGS = -std=c++11

HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../robin-map/include/

.PHONY: build
build:
	$(CXX) $(C_CXX_FLAGS) $(CXX_FLAGS) -o libusearch.so -O3 lib.cpp $(HEADER_INCLUDES) -shared -fPIC

.PHONY: test
test:
	$(CXX) $(C_CXX_FLAGS) $(CXX_FLAGS) test.c -L. -lusearch -Wl,-rpath,. -o test
