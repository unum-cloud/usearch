
export enum MetricKind {
    Unknown = 'unknown',
    Cos = 'cos',
    IP = 'ip',
    L2sq = 'l2sq',
    Haversine = 'haversine',
    Pearson = 'pearson',
    Jaccard = 'jaccard',
    Hamming = 'hamming',
    Tanimoto = 'tanimoto',
    Sorensen = 'sorensen'
}

export enum ScalarKind {
    Unknown = 'unknown',
    F32 = 'f32',
    F64 = 'f64',
    F16 = 'f16',
    I8 = 'i8',
    B1 = 'b1'
}

export type IntOrAlike = number | bigint;
export type Keys = BigUint64Array;
export type Distances = Float32Array;

export type KeyOrKeys = bigint | bigint[] | BigUint64Array;
export type IndicatorOrIndicators = boolean | boolean[];
export type CountOrCounts = bigint | BigUint64Array;
export type VectorOrVectors = Float32Array | Float64Array | Int8Array;

/** Represents a set of search results */
export interface Matches {
    /** Keys of the nearest neighbors found (size: n*k). */
    keys: Keys;
    /** Distances of the nearest neighbors found (size: n*k). */
    distances: Distances;
}

/** Represents a set of batched search results */
export class BatchMatches {
    /** Keys of the nearest neighbors found (size: n*k). */
    keys: Keys;
    /** Distances of the nearest neighbors found (size: n*k). */
    distances: Distances;
    /** Counts of the nearest neighbors found (size: n*k). */
    counts: BigUint64Array;
    /** Limit for search results per query. */
    k: bigint;

    /** Retrieve Matches object at the specified index in the batch. */
    get(i: IntOrAlike): Matches;
}

/** K-Approximate Nearest Neighbors search index. */
export class Index {

    /**
     * Constructs a new index.
     * 
     * @param {IntOrAlike} dimensions
     * @param {MetricKind} metric
     * @param {ScalarKind} quantization
     * @param {IntOrAlike} capacity
     * @param {IntOrAlike} connectivity
     * @param {IntOrAlike} expansion_add
     * @param {IntOrAlike} expansion_search
     * @param {boolean} multi
     */
    constructor(
        dimensions: IntOrAlike,
        metric: MetricKind,
        quantization: ScalarKind,
        capacity: IntOrAlike,
        connectivity: IntOrAlike,
        expansion_add: IntOrAlike,
        expansion_search: IntOrAlike,
        multi: boolean,
    );

    /**
     * Returns the dimensionality of vectors.
     * @return {bigint} The dimensionality of vectors.
     */
    dimensions(): bigint;

    /**
     * Returns the bigint of vectors currently indexed.
     * @return {bigint} The number of vectors currently indexed.
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
    view(path: string): void;

    /** 
     * Add n vectors of dimension d to the index.
     * 
     * @param {KeyOrKeys} keys Input identifiers for every vector.
     * @param {VectorOrVectors} vectors Input matrix, matrix of size n * d.
     */
    add(keys: KeyOrKeys, vectors: VectorOrVectors): void;

    /** 
     * Query n vectors of dimension d to the index. Return at most k vectors for each. 
     * If there are not enough results for a query, the result array is padded with -1s.
     *
     * @param {VectorOrVectors} vectors Input vectors to search, matrix of size n * d.
     * @param {IntOrAlike} k The number of nearest neighbors to search for.
     * @return {Matches | BatchMatches} Search results for one or more queries.
     */
    search(vectors: VectorOrVectors, k: IntOrAlike): Matches | BatchMatches;

    /**
     * Check if one or more entries are contained in the index.
     * @param {KeyOrKeys} keys - Identifier(s) to look up.
     * @return {IndicatorOrIndicators} - Returns true if the key is contained in the index, false otherwise when a single key is provided.
     *                                   Returns an array of booleans corresponding to the presence of each key in the index when an array of keys is provided.
     */
    contains(keys: KeyOrKeys): IndicatorOrIndicators;

    /**
     * Check if one or more entries are contained in the index.
     * @param {KeyOrKeys} keys - Identifier(s) to look up.
     * @return {CountOrCounts} - Number of vectors found per query.
     */
    contains(keys: KeyOrKeys): CountOrCounts;


    /** 
     * Remove a vector from the index.
     * 
     * @param {KeyOrKeys} keys Identifier(s) for every vector to be removed.
     * @return {CountOrCounts} - Number of vectors deleted per query.
     */
    remove(keys: KeyOrKeys): CountOrCounts;

}

/**
 * Performs an exact search on the given dataset to find the best matching vectors for each query.
 * 
 * @param {VectorOrVectors} dataset - The dataset containing vectors to be searched. It should be a flat array representing a matrix of size `n * dimensions`, where `n` is the number of vectors, and `dimensions` is the number of elements in each vector.
 * @param {VectorOrVectors} queries - The queries containing vectors to search for in the dataset. It should be a flat array representing a matrix of size `m * dimensions`, where `m` is the number of query vectors, and `dimensions` is the number of elements in each vector.
 * @param {IntOrAlike} dimensions - The dimensionality of the vectors in both the dataset and the queries. It defines the number of elements in each vector.
 * @param {IntOrAlike} count - The number of nearest neighbors to return for each query. If the dataset contains fewer vectors than the specified count, the result will contain only the available vectors.
 * @param {MetricKind} metric - The distance metric to be used for the search. It should be one of the supported metric strings, for example, "euclidean" for Euclidean distance, "cosine" for Cosine distance, etc.
 * @return {Matches} - Returns a `Matches` object containing the results of the search. The `keys` field contains the indices of the matching vectors in the dataset, the `distances` field contains the distances between the query and the matching vectors, and the `count` field contains the actual number of matches found for each query.
 * 
 * @example
 * const dataset = new VectorOrVectors([1.0, 2.0, 3.0, 4.0]); // Two vectors: [1.0, 2.0] and [3.0, 4.0]
 * const queries = new VectorOrVectors([1.5, 2.5]); // One vector: [1.5, 2.5]
 * const dimensions = BigInt(2);
 * const count = BigInt(1);
 * const metric = "euclidean";
 * 
 * const result = exactSearch(dataset, queries, dimensions, count, metric);
 * // result might be: 
 * // {
 * //    keys: BigUint64Array [ 1n ],
 * //    distances: VectorOrVectors [ some_value ],
 * //    count: 1n
 * // }
 */
export function exactSearch(dataset: VectorOrVectors, queries: VectorOrVectors, dimensions: IntOrAlike, count: IntOrAlike, metric: MetricKind): Matches | BatchMatches;
