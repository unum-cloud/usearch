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
    /// Function type used to filter out keys in results during search.
    /// The filter function should return true to include, and false to skip.
    public typealias FilterFn = (Key) -> Bool

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
    public func add(key: USearchKey, vector: [Float32]) {
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
    public func search(vector: [Float32], count: Int) -> ([Key], [Float]) {
        return search(vector: vector[...], count: count)
    }

    /// Retrieve vectors for a given key.
    /// - Parameter key: Unique identifier for that object.
    /// - Parameter count: For multi-indexes, Number of vectors to retrieve. Defaults to 1.
    /// - Returns: Two-dimensional array of Single-precision vectors.
    /// - Throws: If runs out of memory.
    public func get(key: USearchKey, count: Int = 1) -> [[Float]]? {
        var vector: [Float] = Array(repeating: 0.0, count: Int(self.dimensions) * count)
        let returnedCount = vector.withContiguousMutableStorageIfAvailable { buf in
            guard let baseAddress = buf.baseAddress else { return UInt32(0) }
            return getSingle(
                key: key,
                vector: baseAddress,
                count: CUnsignedInt(count)
            )
        }
        guard let count = returnedCount, count > 0 else { return nil }
        return stride(
            from: 0,
            to: Int(count) * Int(self.dimensions),
            by: Int(self.dimensions)
        ).map {
            Array(vector[$0 ..< $0 + Int(self.dimensions)])
        }
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
    public func add(key: Key, vector: [Float64]) {
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
    public func search(vector: [Float64], count: Int) -> ([Key], [Float]) {
        search(vector: vector[...], count: count)
    }

    /// Retrieve vectors for a given key.
    /// - Parameter key: Unique identifier for that object.
    /// - Parameter count: For multi-indexes, Number of vectors to retrieve. Defaults to 1.
    /// - Returns: Two-dimensional array of Double-precision vectors.
    /// - Throws: If runs out of memory.
    public func get(key: USearchKey, count: Int = 1) -> [[Float64]]? {
        var vector: [Float64] = Array(repeating: 0.0, count: Int(self.dimensions) * count)
        let count = vector.withContiguousMutableStorageIfAvailable { buf in
            guard let baseAddress = buf.baseAddress else { return UInt32(0) }
            return getDouble(
                key: key,
                vector: baseAddress,
                count: CUnsignedInt(count)
            )
        }
        guard let count = count, count > 0 else { return nil }
        return stride(
            from: 0,
            to: Int(count) * Int(self.dimensions),
            by: Int(self.dimensions)
        ).map {
            Array(vector[$0 ..< $0 + Int(self.dimensions)])
        }
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Single-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Parameter filter: Closure used to determine whether to skip a key in the results.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func filteredSearch(vector: ArraySlice<Float32>, count: Int, filter: @escaping FilterFn) -> ([Key], [Float])
    {
        var matches: [Key] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            filteredSearchSingle(
                vector: $0.baseAddress!,
                count:
                    CUnsignedInt(count),
                filter: filter,
                keys: &matches,
                distances: &distances
            )
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Single-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Parameter filter: Closure used to determine whether to skip a key in the results.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func filteredSearch(vector: [Float32], count: Int, filter: @escaping FilterFn) -> ([Key], [Float]) {
        filteredSearch(vector: vector[...], count: count, filter: filter)
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Double-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Parameter filter: Closure used to determine whether to skip a key in the results.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func filteredSearch(vector: ArraySlice<Float64>, count: Int, filter: @escaping FilterFn) -> ([Key], [Float])
    {
        var matches: [Key] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            filteredSearchDouble(
                vector: $0.baseAddress!,
                count:
                    CUnsignedInt(count),
                filter: filter,
                keys: &matches,
                distances: &distances
            )
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }

    /// Approximate nearest neighbors search.
    /// - Parameter vector: Double-precision query vector.
    /// - Parameter count: Upper limit on the number of matches to retrieve.
    /// - Parameter filter: Closure used to determine whether to skip a key in the results.
    /// - Returns: Labels and distances to closest approximate matches in decreasing similarity order.
    /// - Throws: If runs out of memory.
    public func filteredSearch(vector: [Float64], count: Int, filter: @escaping FilterFn) -> ([Key], [Float]) {
        filteredSearch(vector: vector[...], count: count, filter: filter)
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
        public func add(key: Key, vector: [Float16]) {
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
        public func search(vector: [Float16], count: Int) -> ([Key], [Float]) {
            search(vector: vector[...], count: count)
        }

        /// Retrieve vectors for a given key.
        /// - Parameter key: Unique identifier for that object.
        /// - Parameter count: For multi-indexes, Number of vectors to retrieve. Defaults to 1.
        /// - Returns: Two-dimensional array of Half-precision vectors.
        /// - Throws: If runs out of memory.
        @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
        public func get(key: USearchKey, count: Int = 1) -> [[Float16]]? {
            var vector: [Float16] = Array(repeating: 0.0, count: Int(self.dimensions) * count)
            let count = vector.withContiguousMutableStorageIfAvailable { buf in
                guard let baseAddress = buf.baseAddress else { return UInt32(0) }
                return getSingle(
                    key: key,
                    vector: baseAddress,
                    count: CUnsignedInt(count)
                )
            }
            guard let count = count, count > 0 else { return nil }
            return stride(
                from: 0,
                to: Int(count) * Int(self.dimensions),
                by: Int(self.dimensions)
            ).map {
                Array(vector[$0 ..< $0 + Int(self.dimensions)])
            }
        }

    #endif

    /// Number of vectors in the index.
    public var count: Int { return Int(length) }
}
