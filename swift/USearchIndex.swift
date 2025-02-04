//
//  USearchIndex.swift
//
//
//  Created by Ash Vardanian on 5/11/23.
//

import Foundation
import usearch_cxx

enum USearchScalar: UInt {
    case f32
    case f16
    case f64
    case i8
    case b1
    case bf16
}

enum USearchMetric: UInt {
    case unknown
    case ip
    case cos
    case l2sq
    case pearson
    case haversine
    case divergence
    case jaccard
    case hamming
    case tanimoto
    case sorensen
}

enum USearchError: Int, Error {
    case unsupportedMetric
    case addError
    case findError
    case allocationError
    case removeError
    case renameError
    case pathNotUTF8Encodable
    case saveError
    case loadError
    case viewError
}

typealias USearchKey = UInt64
typealias USearchFilterFn = (USearchKey) -> Bool

typealias metric_kind_t = unum.usearch.metric_kind_t
typealias scalar_kind_t = unum.usearch.scalar_kind_t
typealias index_dense_t = unum.usearch.index_dense_t
typealias state_result_t = unum.usearch.index_dense_t.state_result_t

extension USearchMetric {
    fileprivate func toNative() -> metric_kind_t {
        switch self {
        case .ip:
            return metric_kind_t.ip_k
        case .cos:
            return metric_kind_t.cos_k
        case .l2sq:
            return metric_kind_t.l2sq_k
        case .hamming:
            return metric_kind_t.hamming_k
        case .haversine:
            return metric_kind_t.haversine_k
        case .divergence:
            return metric_kind_t.divergence_k
        case .jaccard:
            return metric_kind_t.jaccard_k
        case .pearson:
            return metric_kind_t.pearson_k
        case .sorensen:
            return metric_kind_t.sorensen_k
        case .tanimoto:
            return metric_kind_t.tanimoto_k
        default:
            return metric_kind_t.unknown_k
        }
    }
}

extension USearchScalar {
    fileprivate func toNative() -> scalar_kind_t {
        switch self {
        case .i8:
            return scalar_kind_t.i8_k
        case .f16:
            return scalar_kind_t.f16_k
        case .bf16:
            return scalar_kind_t.bf16_k
        case .f32:
            return scalar_kind_t.f32_k
        case .f64:
            return scalar_kind_t.f64_k
        default:
            return scalar_kind_t.unknown_k
        }
    }
}


@available(iOS 13.0, macOS 11.0, tvOS 13.0, watchOS 6.0, *)
class USearchIndex: NSObject {

    private var nativeIndex: index_dense_t

    var isEmpty: Bool {
        return nativeIndex.size() != 0
    }

    override var description: String {
        return "USearchIndex(dimensions: \(dimensions), connectivity: \(connectivity), length: \(length), capacity: \(capacity), isEmpty: \(isEmpty))"
    }

    let dimensions: UInt32
    let connectivity: UInt32
    let expansionAdd: UInt32
    let expansionSearch: UInt32

    let length: UInt32
    let capacity: UInt32


    private init(native: consuming index_dense_t) {
        nativeIndex = native
        dimensions = UInt32(nativeIndex.dimensions())
        connectivity = UInt32(nativeIndex.connectivity())
        expansionAdd = UInt32(nativeIndex.expansion_add())
        expansionSearch = UInt32(nativeIndex.expansion_search())
        length = UInt32(nativeIndex.size())
        capacity = UInt32(nativeIndex.capacity())
        super.init()
    }


    /**
     * @brief Initializes a new index.
     * @param metric The distance function to compare the dis-similarity of vectors.
     * @param dimensions The number of dimensions planned for this index.
     * @param connectivity Number of connections per node in the proximity graph.
     * Higher connectivity improves quantization, increases memory usage, and reduces construction speed.
     * @param quantization Quantization of internal vector representations. Lower quantization means higher speed.
     */
    static func make(metric: USearchMetric, dimensions: UInt32, connectivity: UInt32, quantization: USearchScalar) throws -> USearchIndex {
        return try make(metric: metric, dimensions: dimensions, connectivity: connectivity, quantization: quantization, multi: false)
    }

    /**
     * @brief Initializes a new index.
     * @param metric The distance function to compare the dis-similarity of vectors.
     * @param dimensions The number of dimensions planned for this index.
     * @param connectivity Number of connections per node in the proximity graph.
     * Higher connectivity improves quantization, increases memory usage, and reduces construction speed.
     * @param quantization Quantization of internal vector representations. Lower quantization means higher speed.
     * @param multi Enables indexing multiple vectors per key when true.
     */
    static func make(metric: USearchMetric, dimensions: UInt32, connectivity: UInt32, quantization: USearchScalar, multi: Bool) throws -> USearchIndex {
        var config = unum.usearch.index_dense_config_t.init(Int(connectivity), 0, 0)
        config.multi = multi

        let nativeMetric = unum.usearch.metric_punned_t.init(Int(dimensions), metric.toNative(), quantization.toNative())
        if nativeMetric.missing() {
            throw USearchError.unsupportedMetric
        }

        let stateResult: unum.usearch.index_dense_gt.state_result_t = unum.usearch.index_dense_t.make(nativeMetric, config)
        if !stateResult {
            throw USearchError.allocationError
        }

        let index = stateResult.index
        return USearchIndex(native: index)
    }


    /**
     * @brief Pre-allocates space in the index for the given number of vectors.
     */
    func reserve(_ count: UInt32) throws {
        if !nativeIndex.try_reserve(Int(count)) {
            throw USearchError.allocationError
        }
    }

    /**
     * @brief Adds a labeled vector to the index.
     * @param vector Single-precision vector.
     */
    func addSingle(key: USearchKey, vector: [Float32]) throws {
        let result = nativeIndex.add(key, vector)
        if !result {
            throw USearchError.addError
        }
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Single-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    func searchSingle(vector: [Float32], count: UInt32, keys: UnsafeMutablePointer<USearchKey>?, distances: UnsafeMutablePointer<Float32>?) throws -> UInt32 {
        let result = nativeIndex.search(vector, Int(count))
        if !result {
            throw USearchError.findError
        }
        let found = result.dump_to(keys, distances)
        return UInt32(found)
    }

    /**
    * @brief Retrieves a labeled single-precision vector from the index.
    * @param vector A buffer to store the vector.
    * @param count For multi-indexes, the number of vectors to retrieve.
    * @return Number of vectors exported to `vector`.
    */
    func getSingle(key: USearchKey, vector: UnsafeMutableRawPointer, count: UInt32) throws -> UInt32 {
        let result = nativeIndex.get(key, vector.assumingMemoryBound(to: Float32.self), Int(count))
        return UInt32(result)
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Single-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param filter Closure called for each key, determining whether to include or
     *               skip key in the results.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    func filteredSearchSingle(vector: [Float32], count: UInt32, filter: USearchFilterFn?, keys: UnsafeMutablePointer<USearchKey>?, distances: UnsafeMutablePointer<Float32>?) throws -> UInt32 {
        let filterBlock: USearchFilterFn? = filter
        let result = nativeIndex.filtered_search(vector, Int(count), filterBlock)

        if !result {
            throw USearchError.findError
        }

        let found = result.dump_to(keys: keys, distances: distances)
        return UInt32(found)
    }

    /**
     * @brief Adds a labeled vector to the index.
     * @param vector Double-precision vector.
     */
    func addDouble(key: USearchKey, vector: [Float64]) throws {
        let result = nativeIndex.add(key, vector)
        if !result {
            throw USearchError.addError
        }
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Double-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    func searchDouble(vector: [Float64], count: UInt32, keys: UnsafeMutablePointer<USearchKey>?, distances: UnsafeMutablePointer<Float32>?) throws -> UInt32 {
        let result = nativeIndex.search(vector, Int(count))
        if !result {
            throw USearchError.findError
        }
        let found = result.dump_to(keys, distances)
        return UInt32(found)
    }

    /**
    * @brief Retrieves a labeled double-precision vector from the index.
    * @param vector A buffer to store the vector.
    * @param count For multi-indexes, the number of vectors to retrieve.
    * @return Number of vectors exported to `vector`.
    */
    func getDouble(key: USearchKey, vector: UnsafeMutableRawPointer, count: UInt32) throws -> UInt32 {
        let result = nativeIndex.get(key, vector.assumingMemoryBound(to: Float64.self), Int(count))
        return UInt32(result)
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Double-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param filter Closure called for each key, determining whether to include or
     *               skip key in the results.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    func filteredSearchDouble(vector: [Float64], count: UInt32, filter: USearchFilterFn?, keys: UnsafeMutablePointer<USearchKey>?, distances: UnsafeMutablePointer<Float32>?) throws -> UInt32 {
        let filterBlock: USearchFilterFn? = filter
        let result = nativeIndex.filtered_search(vector, Int(count), filterBlock)

        if !result {
            throw USearchError.findError
        }

        let found = result.dump_to(keys: keys, distances: distances)
        return UInt32(found)
    }
    /**
     * @brief Adds a labeled vector to the index.
     * @param vector Half-precision vector.
     */
    func addHalf(key: USearchKey, vector: Data) throws {
        let result = nativeIndex.add(key, vector)
        if !result {
            throw USearchError.addError
        }
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Half-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    func searchHalf(vector: Data, count: UInt32, keys: UnsafeMutablePointer<USearchKey>?, distances: UnsafeMutablePointer<Float32>?) throws -> UInt32 {
        let result = nativeIndex.search(vector, Int(count))
        if !result {
            throw USearchError.findError
        }
        let found = result.dump_to(keys: keys, distances: distances)
        return UInt32(found)
    }

    /**
    * @brief Retrieves a labeled half-precision vector from the index.
    * @param vector A buffer to store the vector.
    * @param count For multi-indexes, the number of vectors to retrieve.
    * @return Number of vectors exported to `vector`.
    */
    func getHalf(key: USearchKey, vector: UnsafeMutableRawPointer, count: UInt32) throws -> UInt32 {
        let result = nativeIndex.get(key, vector.assumingMemoryBound(to: Float16.self), Int(count))
        return UInt32(result)
    }

    func contains(key: USearchKey) throws -> Bool {
        return nativeIndex.contains(key)
    }

    func count(key: USearchKey) throws -> UInt32 {
        return UInt32(nativeIndex.count(key))
    }

    func remove(key: USearchKey) throws {
        let result = nativeIndex.remove(key)
        if !result {
            throw USearchError.removeError
        }
    }

    func rename(from key: USearchKey, to newKey: USearchKey) throws {
        let result = nativeIndex.rename(key, newKey)
        if !result {
            throw USearchError.renameError
        }
    }


    /**
     * @brief Saves pre-constructed index to disk.
     */
    func save(path: String) throws {
        guard let cPath = path.cString(using: .utf8) else {
            throw USearchError.pathNotUTF8Encodable
        }
        let result = nativeIndex.save(path: cPath)
        if !result {
            throw USearchError.saveError
        }
    }

    /**
     * @brief Loads a pre-constructed index from index.
     */
    func load(path: String) throws {
        guard let cPath = path.cString(using: .utf8) else {
            throw USearchError.pathNotUTF8Encodable
        }
        let result = nativeIndex.load(path: cPath)
        if !result {
            throw USearchError.loadError
        }
    }

    /**
     * @brief Views a pre-constructed index from disk without loading it into RAM.
     *        Allows working with larger-than memory indexes and saving scarce
     *        memory on device in read-only workloads.
     */
    func view(path: String) throws {
        guard let cPath = path.cString(using: .utf8) else {
            throw USearchError.pathNotUTF8Encodable
        }
        let result = nativeIndex.view(path: cPath)
        if !result {
            throw USearchError.viewError
        }
    }

    /**
     * @brief Removes all the data from index, while preserving the settings.
     */
    func clear() throws {
        nativeIndex.clear()
    }
}
