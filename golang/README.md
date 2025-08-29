# USearch for Go

## Installation

### Linux

Download and install the Debian package from the latest release.
Substitute `<release_tag>`, `<arch>`, and `<usearch_version>` with your settings.

```sh
wget https://github.com/unum-cloud/usearch/releases/download/<release_tag>/usearch_linux_<arch>_<usearch_version>.deb
dpkg -i usearch_linux_<arch>_<usearch_version>.deb
```

### Windows

Run the `winlibinstaller.bat` script from the main repository in the folder where you will run `go run`.
This will install the USearch library and include it in the same folder where the script was run.

```sh
.\usearch\winlibinstaller.bat
```

### macOS

Download and unpack the zip archive from the latest release.
Move the USearch library and the include file to their respective folders.

```sh
wget https://github.com/unum-cloud/usearch/releases/download/<release_tag>/usearch_macos_<arch>_<usearch_version>.zip
unzip usearch_macos_<arch>_<usearch_version>.zip
sudo mv libusearch_c.dylib /usr/local/lib && sudo mv usearch.h /usr/local/include
```

## Quickstart

1. Create a `go.mod` file:

```
module usearch_example

go <go_version>
```

2. Create an `example.go`:

```go
package main

import (
	"fmt"
	usearch "github.com/unum-cloud/usearch/golang"
)

func main() {

   	// Create Index
   	vectorSize := 3
   	vectorsCount := 100
   	conf := usearch.DefaultConfig(uint(vectorSize))
   	index, err := usearch.NewIndex(conf)
   	if err != nil {
   		panic("Failed to create Index")
   	}
   	defer index.Destroy()

   	// Add to Index
   	err = index.Reserve(uint(vectorsCount))
   	for i := 0; i < vectorsCount; i++ {
   		err = index.Add(usearch.Key(i), []float32{float32(i), float32(i + 1), float32(i + 2)})
      	if err != nil {
      		panic("Failed to add")
      	}
   	}

   	// Search
   	keys, distances, err := index.Search([]float32{0.0, 1.0, 2.0}, 3)
   	if err != nil {
    	panic("Failed to search")
   	}
   	fmt.Println(keys, distances)
}
```

3. Get USearch:

```sh
go get github.com/unum-cloud/usearch/golang
```

4. Run:

```sh
go run example.go
```

## Serialization

To save and load the index from disk, use the following methods:

```go
err := index.Save("index.usearch")
if err != nil {
    panic("Failed to save index")
}

err = index.Load("index.usearch")
if err != nil {
    panic("Failed to load index")
}

err = index.View("index.usearch")
if err != nil {
    panic("Failed to view index")
}
```
