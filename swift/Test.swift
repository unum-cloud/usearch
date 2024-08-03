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

        let fetched: [[Float]]? = index.get(key: 42)
        assert(fetched?[0] == vectorA)

        assert(index.contains(key: 42))
        assert(index.count(key: 42) == 1)
        assert(index.count(key: 49) == 0)
        index.rename(from: 42, to: 49)
        assert(index.count(key: 49) == 1)

        let refetched: [[Float]]? = index.get(key: 49)
        assert(refetched?[0] == vectorA)
        let stale: [[Float]]? = index.get(key: 42)
        assert(stale == nil)

        index.remove(key: 49)
        assert(index.count(key: 49) == 0)
    }

    func testUnitMulti() throws {
        let index = USearchIndex.make(
            metric: USearchMetric.l2sq,
            dimensions: 4,
            connectivity: 8,
            quantization: USearchScalar.F32,
            multi: true
        )
        let vectorA: [Float32] = [0.3, 0.5, 1.2, 1.4]
        let vectorB: [Float32] = [0.4, 0.2, 1.2, 1.1]
        index.reserve(2)

        // Adding a slice
        index.add(key: 42, vector: vectorA[...])

        // Adding a vector
        index.add(key: 42, vector: vectorB)

        let results = index.search(vector: vectorA, count: 10)
        assert(results.0[0] == 42)

        let fetched: [[Float]]? = index.get(key: 42, count: 2)
        assert(fetched?.contains(vectorA) == true)
        assert(fetched?.contains(vectorB) == true)

        assert(index.contains(key: 42))
        assert(index.count(key: 42) == 2)
        assert(index.count(key: 49) == 0)
        index.rename(from: 42, to: 49)
        assert(index.count(key: 49) == 2)

        let refetched: [[Float]]? = index.get(key: 49, count: 2)
        assert(refetched?.contains(vectorA) == true)
        assert(refetched?.contains(vectorB) == true)
        let stale: [[Float]]? = index.get(key: 42)
        assert(stale == nil)

        index.remove(key: 49)
        assert(index.count(key: 49) == 0)
    }
}
