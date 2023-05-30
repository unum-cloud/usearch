#pragma once
#include <stdlib.h>

typedef struct {
    int* Labels;
    int LabelsLen;
    const char* Error;
} SearchResults;