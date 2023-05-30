#pragma once
#include <stdlib.h>

typedef struct {
    int* Labels;
    float* Distances;
    int Len;
    char* Error;
} SearchResults;
