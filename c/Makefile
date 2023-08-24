CC = gcc
CXX = g++

C_FLAGS = -std=c99
CXX_FLAGS = -std=c++11

CXX_FLAGS += -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas -pedantic

HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../robin-map/include/

.PHONY: libusearch_c.a
libusearch_c.a:
	$(CXX) $(CXX_FLAGS) -o libusearch_c.a -O3 lib.cpp $(HEADER_INCLUDES) -static

.PHONY: libusearch_c.so
libusearch_c.so:
	$(CXX) $(CXX_FLAGS) -o libusearch_c.so -O3 lib.cpp $(HEADER_INCLUDES) -shared -fPIC

.PHONY: libusearch_c.dylib
libusearch_c.dylib:
	$(CXX) $(CXX_FLAGS) -o libusearch_c.dylib -O3 lib.cpp $(HEADER_INCLUDES) -dynamiclib -fPIC

.PHONY: test_c
test_c:
	$(CC) $(C_FLAGS) test.c -L. -lusearch_c -Wl,-rpath,. -o test
