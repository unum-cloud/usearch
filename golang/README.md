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
	"github.com/unum-cloud/usearch/golang"
)

func main() {
	conf := usearch.DefaultConfig(128)
	index := usearch.NewIndex(conf)
	v := make([]float32, 128)
	index.Add(42, v)
	results := index.Search(v, 1)
}
```
