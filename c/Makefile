CC = gcc-12

C_FLAGS = -std=c99

C_CXX_FLAGS = -Wall -Wextra -Wno-conversion -Wno-unknown-pragmas
C_CXX_FLAGS += -O3 -march=native

HEADER_INCLUDES = -I.  -I ../include/  -I ../fp16/include/ -I ../robin-map/include/

.PHONY: build
build:
	$(CC) $(C_FLAGS) $(C_CXX_FLAGS) -o libusearch.so -O3 lib.cpp $(HEADER_INCLUDES) -shared -fPIC

.PHONY: test
test:
	$(CC) $(C_FLAGS) $(C_CXX_FLAGS) test.c -L. -lusearch -Wl,-rpath,. -o test
