//
//  Index+Sugar.swift
//  
//
//  Created by Ashot Vardanian on 5/11/23.
//

extension Index {

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
            addPrecise(label: label, vector: $0.baseAddress!)
        }
    }

    public func search(vector: ArraySlice<Float64>, count: Int) -> ([UInt32], [Float]) {
        var matches: [UInt32] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            searchPrecise(vector: $0.baseAddress!, count: CUnsignedInt(count), labels: &matches, distances: &distances)
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }
    
    @available(macOS 11.0, *)
    public func add(label: UInt32, vector: ArraySlice<Float16>) -> () {
        vector.withContiguousStorageIfAvailable { buffer in
            addImprecise(label: label, vector: buffer.baseAddress!)
        }
    }

    @available(macOS 11.0, *)
    public func search(vector: ArraySlice<Float16>, count: Int) -> ([UInt32], [Float]) {
        var matches: [UInt32] = Array(repeating: 0, count: count)
        var distances: [Float] = Array(repeating: 0, count: count)
        let results = vector.withContiguousStorageIfAvailable {
            searchImprecise(vector: $0.baseAddress!, count: CUnsignedInt(count), labels: &matches, distances: &distances)
        }
        matches.removeLast(count - Int(results!))
        distances.removeLast(count - Int(results!))
        return (matches, distances)
    }
    
    public var count: Int { get { return Int(length) } }
    
}
