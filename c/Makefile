CC = gcc
CXX = g++

C_FLAGS = -std=c99
CXX_FLAGS = -std=c++11

CXX_FLAGS += -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas -O3 -march=native -pedantic

HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../robin-map/include/

.PHONY: build
build:
	$(CXX) $(CXX_FLAGS) -o libusearch.so -O3 lib.cpp $(HEADER_INCLUDES) -shared -fPIC

.PHONY: test
test:
	$(CC) $(C_FLAGS) test.c -L. -lusearch -Wl,-rpath,. -o test
