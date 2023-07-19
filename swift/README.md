# USearch for Swift

## Installation

```txt
https://github.com/unum-cloud/usearch
```

## Quickstart

```swift
let index = Index.l2sq(dimensions: 3, connectivity: 8)
let vectorA: [Float32] = [0.3, 0.5, 1.2]
let vectorB: [Float32] = [0.4, 0.2, 1.2]
index.add(label: 42, vector: vectorA[...])
index.add(label: 43, vector: vectorB[...])

let results = index.search(vector: vectorA[...], count: 10)
assert(results.0[0] == 42)
```
