# USearch for Swift

## Installation

```txt
https://github.com/unum-cloud/usearch
```

## Quickstart

```swift
let index = USearchIndex.make(metric: .Cos, dimensions: 3, connectivity: 8)
let vectorA: [Float32] = [0.3, 0.5, 1.2] // `Float32` and `Float64` are always supported
let vectorB: [Float32] = [0.4, 0.2, 1.2] // `Float16` supports on the OS & hardware version
index.add(key: 42, vector: vectorA) // Pass full arrays or slices
index.add(key: 43, vector: vectorB)

let results = index.search(vector: vectorA, count: 10)
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
            let _ = index.add(key: 10, vector: coordiantes)            
            VStack {
                Text("USearch index contains \(index.length) vectors")
                Spacer()
            }

        }
    }
}
```

You can find a working example of using USearch with SwiftUI maps at [ashvardanian/SwiftVectorSearch](https://github.com/ashvardanian/SwiftVectorSearch).

[![](https://media.githubusercontent.com/media/ashvardanian/SwiftVectorSearch/main/USearch%2BSwiftUI.gif)](https://github.com/ashvardanian/SwiftVectorSearch)
