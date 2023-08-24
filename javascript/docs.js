/** Search result object. */
class Matches {
    /**
     * @param {BigUint64Array} keys - The keys of the nearest neighbors found.
     * @param {Float32Array} distances - The distances of the nearest neighbors found.
     * @param {bigint} count - The count of nearest neighbors found.
     */
    constructor(keys, distances, count) {
        this.keys = keys;
        this.distances = distances;
        this.count = count;
    }
}

/** K-Approximate Nearest Neighbors search index. */
class Index {
    /**
     * Constructs a new index.
     * 
     * @param {bigint} dimensions
     * @param {string} metric
     * @param {string} quantization
     * @param {bigint} capacity
     * @param {bigint} connectivity
     * @param {bigint} expansion_add
     * @param {bigint} expansion_search
     */
    constructor(
        dimensions,
        metric,
        quantization,
        capacity,
        connectivity,
        expansion_add,
        expansion_search
    ) {}

    /**
     * Returns the dimensionality of vectors.
     * @return {bigint} The dimensionality of vectors.
     */
    dimensions() {}

    /**
     * Returns the bigint of vectors currently indexed.
     * @return {bigint} The bigint of vectors currently indexed.
     */
    size() {}

    /**
     * Returns index capacity.
     * @return {bigint} The capacity of index.
     */
    capacity() {}

    /**
     * Returns connectivity.
     * @return {bigint} The connectivity of index.
     */
    connectivity() {}

    /** 
     * Write index to a file.
     * @param {string} path File path to write.
     */
    save(path) {}

    /** 
     * Load index from a file.
     * @param {string} path File path to read.
     */
    load(path) {}

    /** 
     * View index from a file, without loading into RAM.
     * @param {string} path File path to read.
     */
    view(path) {}

    /** 
     * Add n vectors of dimension d to the index.
     * 
     * @param {bigint | bigint[]} keys Input identifiers for every vector.
     * @param {Float32Array | Float32Array[]} mat Input matrix, matrix of size n * d.
     */
    add(keys, mat) {}

    /** 
     * Query n vectors of dimension d to the index. Return at most k vectors for each. 
     * If there are not enough results for a query, the result array is padded with -1s.
     *
     * @param {Float32Array} mat Input vectors to search, matrix of size n * d.
     * @param {bigint} k The bigint of nearest neighbors to search for.
     * @return {Matches} Output of the search result.
     */
    search(mat, k) {}

    /** 
     * Check if an entry is contained in the index.
     * 
     * @param {bigint} key Identifier to look up.
     */
    contains(key) {}

    /** 
     * Remove a vector from the index.
     * 
     * @param {bigint} key Input identifier for every vector to be removed.
     */
    remove(key) {}
}
