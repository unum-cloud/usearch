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
index.add(key: 42, vector: vectorA[...])
index.add(key: 43, vector: vectorB[...])

let results = index.search(vector: vectorA[...], count: 10)
assert(results.0[0] == 42)
```

If using in a SwiftUI application, make sure to annulate the void responses:

```swift
import SwiftUI
import USearch

@main
struct USearchMobileApp: App {
    var body: some Scene {
        WindowGroup {
            let index = USearchIndex.make(metric: .IP, dimensions: 2, connectivity: 16, quantization: .F32)
            let _ = index.reserve(10)
            let coordiantes: Array<Float32> = [40.177200, 44.503490]
            let _ = index.add(key: 10, vector: coordiantes[...])            
            VStack {
                Text("USearch index contains \(index.length) vectors")
                Spacer()
            }

        }
    }
}
```
