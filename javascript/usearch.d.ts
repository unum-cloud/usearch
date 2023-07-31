
/** Search result object. */
export interface SearchResults {
    /** The labels of the nearest neighbors found, size n*k. */
    labels: BigUint64Array,
    /** The disances of the nearest negihbors found, size n*k. */
    distances: Float32Array,
    /** The disances of the nearest negihbors found, size n*k. */
    count: number
}

/** K-Approximate Nearest Neighbors search index. */
export class Index {
    
    /**
     * Constructs a new index.
     * 
     * @param {number} dimensions
     * @param {string} metric
     * @param {string} quantization
     * @param {number} capacity
     * @param {number} connectivity
     * @param {number} expansion_add
     * @param {number} expansion_search
     */
    constructor(...args);

    /**
     * Returns the dimensionality of vectors.
     * @return {number} The dimensionality of vectors.
     */
    dimensions(): number;

    /**
     * Returns the number of vectors currently indexed.
     * @return {number} The number of vectors currently indexed.
     */
    size(): number;

    /**
     * Returns index capacity.
     * @return {numbers} The capacity of index.
     */
    capacity(): number;
    
    /**
     * Returns connectivity.
     * @return {number} The connectivity of index.
     */
    connectivity(): number;

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
     * @param {number | number[]} keys Input identifiers for every vector.
     * @param {Float32Array | Float32Array[]} mat Input matrix, matrix of size n * d.
     */
    add(keys: number | number[], mat: Float32Array | Float32Array[]): void;

    /** 
     * Query n vectors of dimension d to the index. Return at most k vectors for each. 
     * If there are not enough results for a query, the result array is padded with -1s.
     *
     * @param {Float32Array} mat Input vectors to search, matrix of size n * d.
     * @param {number} k The number of nearest neighbors to search for.
     * @return {SearchResults} Output of the search result.
     */
    search(mat: Float32Array, k: number): SearchResults;
}