//
//  Index+Sugar.swift
//
//
//  Created by Ash Vardanian on 5/11/23.
//

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
extension USearchIndex {
    public typealias Key = USearchKey
    public typealias Metric = USearchMetric
    public typealias Scalar = USearchScalar

    /// Adds a labeled vector to the index.
    /// - Parameter key: Unique identifier for that object.
    /// - Parameter vector: Single-precision vector.
    /// - Throws: If runs out of memory.
    public func add(key: USearchKey, vector: ArraySlice<Float32>) {
        vector.withContiguousStorageIfAvailable {
            addSingle(key: key, vector: $0.baseAddress!)
        }
    }

    /// Adds a labeled vector to the index.
    /// - Parameter key: Unique identifier for that object.
    /// - Parameter vector: Single-precision vector.
    /// - Throws: If runs out of memory.
    public func add(key: USearchKey, vector: Array<Float32>) {
        add(key: key, vector: vector[...])
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Single-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func search(vector: ArraySlice<Float32>, count: Int) -> ([Key], [Float]) {
        var matches: [Key] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            searchSingle(vector: $0.baseAddress!, count: CUnsignedInt(count), keys: &matches, distances: &distances)
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Single-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func search(vector: Array<Float32>, count: Int) -> ([Key], [Float]) {
        return search(vector: vector[...], count: count)
    }

    /// Adds a labeled vector to the index.
    /// - Parameter key: Unique identifier for that object.
    /// - Parameter vector: Double-precision vector.
    /// - Throws: If runs out of memory.
    public func add(key: Key, vector: ArraySlice<Float64>) {
        vector.withContiguousStorageIfAvailable {
            addDouble(key: key, vector: $0.baseAddress!)
        }
    }

    /// Adds a labeled vector to the index.
    /// - Parameter key: Unique identifier for that object.
    /// - Parameter vector: Double-precision vector.
    /// - Throws: If runs out of memory.
    public func add(key: Key, vector: Array<Float64>) {
        add(key: key, vector: vector[...])
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Double-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func search(vector: ArraySlice<Float64>, count: Int) -> ([Key], [Float]) {
        var matches: [Key] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            searchDouble(vector: $0.baseAddress!, count: CUnsignedInt(count), keys: &matches, distances: &distances)
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Double-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func search(vector: Array<Float64>, count: Int) -> ([Key], [Float]) {
        search(vector: vector[...], count: count)
    }

    #if arch(arm64)

        /// Adds a labeled vector to the index.
        /// - Parameter key: Unique identifier for that object.
        /// - Parameter vector: Half-precision vector.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func add(key: Key, vector: ArraySlice<Float16>) {
            vector.withContiguousStorageIfAvailable { buffer in
                addHalf(key: key, vector: buffer.baseAddress!)
            }
        }

        /// Adds a labeled vector to the index.
        /// - Parameter key: Unique identifier for that object.
        /// - Parameter vector: Half-precision vector.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func add(key: Key, vector: Array<Float16>) {
            add(key: key, vector: vector[...])
        }

        /// Approximate nearest neighbors search.
        /// - Parameter vector: Half-precision query vector.
        /// - Parameter count: Upper limit on the number of matches to retrieve.
        /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func search(vector: ArraySlice<Float16>, count: Int) -> ([Key], [Float]) {
            var matches: [Key] = Array(repeating: 0, count: count)
            var distances: [Float] = Array(repeating: 0, count: count)
            let results = vector.withContiguousStorageIfAvailable {
                searchHalf(vector: $0.baseAddress!, count: CUnsignedInt(count), keys: &matches, distances: &distances)
            }
            matches.removeLast(count - Int(results!))
            distances.removeLast(count - Int(results!))
            return (matches, distances)
        }

        /// Approximate nearest neighbors search.
        /// - Parameter vector: Half-precision query vector.
        /// - Parameter count: Upper limit on the number of matches to retrieve.
        /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func search(vector: Array<Float16>, count: Int) -> ([Key], [Float]) {
            search(vector: vector[...], count: count)
        }

    #endif

    /// Number of vectors in the index.
    public var count: Int { return Int(length) }
}
