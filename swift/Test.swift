//
//  File.swift
//
//
//  Created by Ash Vardanian on 5/11/23.
//

import Foundation
import USearch
import XCTest

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
class Test: XCTestCase {
    func testUnit() throws {
        let index = USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 4,
            connectivity: 8,
            quantization: USearchScalar.F32
        )
        let vectorA: [Float32] = [0.3, 0.5, 1.2, 1.4]
        let vectorB: [Float32] = [0.4, 0.2, 1.2, 1.1]
        index.reserve(2)

        // Adding a slice
        index.add(key: 42, vector: vectorA[...])

        // Adding a vector
        index.add(key: 43, vector: vectorB)

        let results = index.search(vector: vectorA, count: 10)
        assert(results.0[0] == 42)

        assert(index.contains(key: 42))
        assert(index.count(key: 42) == 1)
        assert(index.count(key: 49) == 0)
        index.rename(from: 42, to: 49)
        assert(index.count(key: 49) == 1)
        index.remove(key: 49)
        assert(index.count(key: 49) == 0)
    }
}
