# USearch for GoLang

## Installation

```golang
import (
	"github.com/unum-cloud/usearch/golang"
)
```

## Quickstart

```golang
package main

import (
	"fmt"
	usearch "github.com/unum-cloud/usearch/golang"
)

func main() {
	// Create Index
	conf := usearch.DefaultConfig(3)
	index,err := usearch.NewIndex(conf)
	defer index.Destroy()
	if err != nil {
		panic("Failed to create Index")
	}
	
	// Add to Index
	err = index.Reserve(3)
	vec := []float32{0.1, 0.2, 0.3}

	err = index.Add(42, vec)
	if err != nil {
		panic("Failed to add")
	}

	// Search
	keys, distances, err := index.Search(vec, 1)
	if err != nil {
		panic("Failed to search")
	}
	fmt.Println(keys, distances)
}
```
