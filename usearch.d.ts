/** Search result object. */
export interface SearchResult {
    /** The labels of the nearest neighbors found, size n*k. */
    labels: BigInt64Array,
    /** The disances of the nearest negihbors found, size n*k. */
    distances: Float32Array,
    /** The disances of the nearest negihbors found, size n*k. */
    count: number
}

export class Index {
    
    constructor(...args);

    /**
     * Returns the dimensionality of verctors.
     * @return {number} The dimensionality of verctors.
     */
    dimensions(): number;

    /**
     * Returns the number of verctors currently indexed.
     * @return {number} The number of verctors currently indexed.
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
     * @param {string} fname File path to write.
     */
    save(fname: string): void;

    /** 
     * Read index from a file.
     * @param {string} fname File path to read.
     * @return {Index} The index read.
     */
    load(fname: string): Index;

    /** 
     * Add n vectors of dimension d to the index.
     * Vectors are implicitly assigned labels size .. size + n - 1
     * @param {number} id Input label
     * @param {Float32Array} arr Input matrix, size n * d
     */
    add(id: number, arr: Float32Array): void;

    /** 
     * Query n vectors of dimension d to the index.
     * Return at most k vectors. If there are not enough results for a
     * query, the result array is padded with -1s.
     *
     * @param {Float32Array} arr Input vectors to search, size n * d.
     * @param {number} k The number of nearest neighbors to search for.
     * @return {SearchResult} Output of the search result.
     */
    search(arr: Float32Array, k: number): SearchResult;
}