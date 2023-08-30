
/** Search result object. */
export interface Matches {
    /** The keys of the nearest neighbors found, size n*k. */
    keys: BigUint64Array,
    /** The distances of the nearest neighbors found, size n*k. */
    distances: Float32Array,
    /** The distances of the nearest neighbors found, size n*k. */
    count: bigint
}

/** K-Approximate Nearest Neighbors search index. */
export class Index {

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
        dimensions: bigint,
        metric: string,
        quantization: string,
        capacity: bigint,
        connectivity: bigint,
        expansion_add: bigint,
        expansion_search: bigint
    );

    /**
     * Returns the dimensionality of vectors.
     * @return {bigint} The dimensionality of vectors.
     */
    dimensions(): bigint;

    /**
     * Returns the bigint of vectors currently indexed.
     * @return {bigint} The bigint of vectors currently indexed.
     */
    size(): bigint;

    /**
     * Returns index capacity.
     * @return {bigints} The capacity of index.
     */
    capacity(): bigint;

    /**
     * Returns connectivity.
     * @return {bigint} The connectivity of index.
     */
    connectivity(): bigint;

    /** 
     * Write index to a file.
     * @param {string} path File path to write.
     */
    save(path: string): void;

    /** 
     * Load index from a file.
     * @param {string} path File path to read.
     */
    load(path: string): void;

    /** 
     * View index from a file, without loading into RAM.
     * @param {string} path File path to read.
     */
    load(path: string): void;

    /** 
     * Add n vectors of dimension d to the index.
     * 
     * @param {bigint | bigint[]} keys Input identifiers for every vector.
     * @param {Float32Array | Float32Array[]} mat Input matrix, matrix of size n * d.
     */
    add(keys: bigint | bigint[], mat: Float32Array | Float32Array[]): void;

    /** 
     * Query n vectors of dimension d to the index. Return at most k vectors for each. 
     * If there are not enough results for a query, the result array is padded with -1s.
     *
     * @param {Float32Array} mat Input vectors to search, matrix of size n * d.
     * @param {bigint} k The bigint of nearest neighbors to search for.
     * @return {Matches} Output of the search result.
     */
    search(mat: Float32Array, k: bigint): Matches;

    /** 
     * Check if an entry is contained in the index.
     * 
     * @param {bigint} key Identifier to look up.
     */
    contains(key: bigint): boolean;

    /** 
     * Remove a vector from the index.
     * 
     * @param {bigint} key Input identifier for every vector to be removed.
     */
    remove(key: bigint): boolean;

}