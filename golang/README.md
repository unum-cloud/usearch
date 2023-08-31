# USearch for GoLang

## Installation

#### Linux

Download and install from the Debian package from the latest release.
Substitute `<release_tag>`, `<arch>`, and `<usearch_version>` with your settings.

```
wget https://github.com/unum-cloud/usearch/releases/download/<release_tag>/usearch_linux_<arch>_<usearch_version>.deb
dpkg -i usearch_<arch>_<usearch_version>.deb
```

#### Windows

Download and run a bat script from the from the main repository.
That will install the USearch library in the same folder` where the script was run.

```
wget https://github.com/unum-cloud/usearch/blob/main/winlibinstaller.bat
.\winlibinstaller.bat
```

#### MacOS

Download and unpack a zip archive from the latest release.
Move the USearch library and the include file to their respective folders.
```
wget https://github.com/unum-cloud/usearch/releases/download/<release_tag>/usearch_macOS_<arch>_<usearch_version>.zip
unzip usearch_macOS_<arch>_<usearch_version>.zip
sudo mv libusearch.so /usr/local/lib && sudo mv usearch.h /usr/local/include

```

## Quickstart

1. Create a `go.mod` file:

	```
	module usearch_example

	go <go_version>
	```

2. Create an `example.go`:

	```golang
	package main

	import (
		"fmt"
		usearch "github.com/unum-cloud/usearch/golang"
	)

	func main() {

		// Create Index
		vector_size := 3
		vectors_count := 100
		conf := usearch.DefaultConfig(uint(vector_size))
		index,err := usearch.NewIndex(conf)
		if err != nil {
			panic("Failed to create Index")
		}
		defer index.Destroy()
		
		// Add to Index
		err = index.Reserve(uint(vectors_count))
		for i := 0; i < vectors_count; i++ {
			err = index.Add(usearch.Key(i), []float32{float32(i), float32(i+1), float32(i+2)})
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

	```
	go get github.com/unum-cloud/usearch/golang
	```

4. Run:

	```
	go run example.go
	```
