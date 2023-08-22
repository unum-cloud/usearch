# USearch for GoLang

## Installation

```golang
import (
	"github.com/unum-cloud/usearch/golang-go"
)
```

## Quickstart

```golang
package main

import (
	"fmt"
	"github.com/unum-cloud/usearch/golang-go"
)

func main() {
	dim := uint(128)
	conf := DefaultConfig(dim)
	ind, err := NewIndex(conf)
	if err != nil {
		panic("Failed to construct the index: %s", err)
	}
	defer ind.Destroy()

	err = ind.Reserve(100)
	if err != nil {
		panic("Failed to reserve capacity: %s", err)
	}

	vec := make([]float32, dim)
	vec[0] = 40.0
	vec[1] = 2.0

	err = ind.Add(42, vec)
	if err != nil {
		panic("Failed to insert: %s", err)
	}

	keys, distances, err := ind.Search(vec, 10)
	if err != nil {
		panic("Failed to search: %s", err)
	}
	if keys[0] != 42 || distances[0] != 0.0 {
		panic("Expected result 42 with distance 0, got key %d with distance %f", keys[0], distances[0])
	}
}
```
