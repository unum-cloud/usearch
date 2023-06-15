//
//  Index+Sugar.swift
//
//
//  Created by Ashot Vardanian on 5/11/23.
//

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
extension USearchIndex {
    /// Adds a labeled vector to the index.
    /// - Parameter label: Unique identifer for that object.
    /// - Parameter vector: Single-precision vector.
    /// - Throws: If runs out of memory.
    public func add(label: UInt32, vector: ArraySlice<Float32>) {
        vector.withContiguousStorageIfAvailable {
            addSingle(label: label, vector: $0.baseAddress!)
        }
    }

    
    /// Approximate nearest neighbors search.
    /// - Parameter vector: Single-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func search(vector: ArraySlice<Float32>, count: Int) -> ([UInt32], [Float]) {
        var matches: [UInt32] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            searchSingle(vector: $0.baseAddress!, count: CUnsignedInt(count), labels: &matches, distances: &distances)
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }

    /// Adds a labeled vector to the index.
    /// - Parameter label: Unique identifer for that object.
    /// - Parameter vector: Double-precision vector.
    /// - Throws: If runs out of memory.
    public func add(label: UInt32, vector: ArraySlice<Float64>) {
        vector.withContiguousStorageIfAvailable {
            addDouble(label: label, vector: $0.baseAddress!)
        }
    }
    
    /// Approximate nearest neighbors search.
    /// - Parameter vector: Double-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func search(vector: ArraySlice<Float64>, count: Int) -> ([UInt32], [Float]) {
        var matches: [UInt32] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            searchDouble(vector: $0.baseAddress!, count: CUnsignedInt(count), labels: &matches, distances: &distances)
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }

    #if arch(arm64)

        /// Adds a labeled vector to the index.
        /// - Parameter label: Unique identifer for that object.
        /// - Parameter vector: Half-precision vector.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func add(label: UInt32, vector: ArraySlice<Float16>) {
            vector.withContiguousStorageIfAvailable { buffer in
                addHalf(label: label, vector: buffer.baseAddress!)
            }
        }

        /// Approximate nearest neighbors search.
        /// - Parameter vector: Half-precision query vector.
        /// - Parameter count: Upper limit on the number of matches to retrieve.
        /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func search(vector: ArraySlice<Float16>, count: Int) -> ([UInt32], [Float]) {
            var matches: [UInt32] = Array(repeating: 0, count: count)
            var distances: [Float] = Array(repeating: 0, count: count)
            let results = vector.withContiguousStorageIfAvailable {
                searchHalf(vector: $0.baseAddress!, count: CUnsignedInt(count), labels: &matches, distances: &distances)
            }
            matches.removeLast(count - Int(results!))
            distances.removeLast(count - Int(results!))
            return (matches, distances)
        }

    #endif

    /// Number of vectors in the index.
    public var count: Int { return Int(length) }
}
