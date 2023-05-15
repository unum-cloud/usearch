//
//  File.swift
//  
//
//  Created by Ashot Vardanian on 5/11/23.
//

import Foundation
import USearch

func testUnit() throws {
    let index = Index.l2(dimensions: 4, connectivity: 8)
    let vectorA: [Float32] = [0.3, 0.5, 1.2, 1.4]
    let vectorB: [Float32] = [0.4, 0.2, 1.2, 1.1]
    index.add(label: 42, vector: vectorA[...])
    index.add(label: 43, vector: vectorB[...])

    let results = index.search(vector: vectorA[...], count: 10)
    assert(results.0[0] == 42)
}
