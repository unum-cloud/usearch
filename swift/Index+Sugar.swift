//
//  Index+Sugar.swift
//  
//
//  Created by Ashot Vardanian on 5/11/23.
//

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
extension USearchIndex {

    public func add(label: UInt32, vector: ArraySlice<Float32>) -> () {
        vector.withContiguousStorageIfAvailable {
            addSingle(label: label, vector: $0.baseAddress!)
        }
    }
    
    
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

    public func add(label: UInt32, vector: ArraySlice<Float64>) -> () {
        vector.withContiguousStorageIfAvailable {
            addDouble(label: label, vector: $0.baseAddress!)
        }
    }

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

    @available(macOS 11.0, iOS 14.0, watchOS 7.0, tvOS 14.0, *)
    public func add(label: UInt32, vector: ArraySlice<Float16>) -> () {
        vector.withContiguousStorageIfAvailable { buffer in
            addHalf(label: label, vector: buffer.baseAddress!)
        }
    }

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

    public var count: Int { get { return Int(length) } }
    
}
