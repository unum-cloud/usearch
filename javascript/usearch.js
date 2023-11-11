const compiled = require('bindings')('usearch');

/**
 * Enumeration representing the various metric kinds used to measure the distance between vectors in the index.
 * @enum {string}
 * @readonly
 */
const MetricKind = {
    Unknown: 'unknown',
    Cos: 'cos',
    IP: 'ip',
    L2sq: 'l2sq',
    Haversine: 'haversine',
    Divergence: 'divergence',
    Pearson: 'pearson',
    Jaccard: 'jaccard',
    Hamming: 'hamming',
    Tanimoto: 'tanimoto',
    Sorensen: 'sorensen'
};

/**
 * Enumeration representing the various scalar kinds used to define the type of scalar values in vectors.
 * @enum {string}
 * @readonly
 */
const ScalarKind = {
    Unknown: 'unknown',
    F32: 'f32',
    F64: 'f64',
    F16: 'f16',
    I8: 'i8',
    B1: 'b1'
};

/**
 * Represents a set of search results.
 */
class Matches {
    /**
     * Constructs a Matches object.
     * 
     * @param {BigUint64Array} keys - The keys of the nearest neighbors found.
     * @param {Float32Array} distances - The distances of the nearest neighbors found.
     */
    constructor(keys, distances) {
        this.keys = keys;
        this.distances = distances;
    }
}

/**
 * Represents a set of batched search results.
 */
class BatchMatches {
    /**
     * Constructs a BatchMatches object.
     * 
     * @param {BigUint64Array} keys - The keys of the nearest neighbors found in the batch.
     * @param {Float32Array} distances - The distances of the nearest neighbors found in the batch.
     * @param {BigUint64Array} counts - The number of neighbors found for each query in the batch.
     * @param {bigint} k - The limit for search results per query in the batch.
     */
    constructor(keys, distances, counts, k) {
        this.keys = keys;
        this.distances = distances;
        this.counts = counts;
        this.k = k;
    }

    /**
     * Retrieves a Matches object at the specified index in the batch.
     * 
     * @param {number} i - The index at which to retrieve the Matches object.
     * @returns {Matches} - A Matches object representing the search results at the specified index in the batch.
     */
    get(i) {
        const index = Number(i) * Number(this.k);
        const count = Number(this.counts[i]);
        const keysSlice = this.keys.slice(index, index + count);
        const distancesSlice = this.distances.slice(index, index + count);
        return new Matches(keysSlice, distancesSlice);
    }
}

function isOneKey(keys) {
    return (!Number.isNaN(keys) && typeof keys === 'number') || typeof keys === 'bigint';
}

function normalizeKeys(keys) {
    if (isOneKey(keys)) {
        keys = BigUint64Array.of(BigInt(keys));
    } else if (Array.isArray(keys)) {
        keys = keys.map(key => {
            if ((typeof key !== 'bigint' && typeof key !== 'number') || Number.isNaN(key))
                throw new Error("All keys must be integers or bigints.");
            return BigInt(key);
        });
        keys = BigUint64Array.from(keys);
    } else if (!(keys instanceof BigUint64Array)) {
        throw new Error("Keys must be a number, bigint, an array of numbers or bigints, or a BigUint64Array.");
    }
    return keys;
}

function isVector(vectors) {
    return vectors instanceof Float32Array || vectors instanceof Float64Array || vectors instanceof Int8Array;
}

function normalizeVectors(vectors, dimensions, targetType = Float32Array) {
    let flattenedVectors;
    if (isVector(vectors)) {
        flattenedVectors = (vectors.constructor === targetType) ? vectors : new targetType(vectors);
    } else if (Array.isArray(vectors)) {
        let totalLength = 0;
        for (const vec of vectors) totalLength += vec.length;

        flattenedVectors = new targetType(totalLength);
        let offset = 0;
        for (const vec of vectors) {
            flattenedVectors.set(vec, offset);
            offset += vec.length;
        }
    } else {
        throw new Error("Vectors must be a TypedArray or an array of arrays.");
    }

    if (flattenedVectors.length % dimensions !== 0)
        throw new Error("The size of the flattened vectors must be a multiple of the dimension of the vectors.");

    return flattenedVectors;
}


class Index {

    /**
     * Constructs a new index.
     * 
     * @param {(number | {dimensions: number, metric: MetricKind = MetricKind.Cos, quantization: ScalarKind = ScalarKind.F32, connectivity: number = 0, expansion_add: number = 0, expansion_search: number = 0, multi: boolean = false})} dimensionsOrConfigs
     * @param {MetricKind} [metric=MetricKind.Cos] - Optional, default is 'cos'.
     * @param {ScalarKind} [quantization=ScalarKind.F32] - Optional, default is 'f32'.
     * @param {number} [connectivity=0] - Optional, default is 0.
     * @param {number} [expansion_add=0] - Optional, default is 0.
     * @param {number} [expansion_search=0] - Optional, default is 0.
     * @param {boolean} [multi=false] - Optional, default is false.
     * @throws Will throw an error if any of the parameters are of incorrect type or invalid value.
     */
    constructor(dimensionsOrConfigs, metric = MetricKind.Cos, quantization = ScalarKind.F32, connectivity = 0, expansion_add = 0, expansion_search = 0, multi = false) {
        let dimensions;
        if (typeof dimensionsOrConfigs === 'object' && dimensionsOrConfigs !== null) {
            // Parameters are provided as an object
            ({ dimensions, metric = MetricKind.Cos, quantization = ScalarKind.F32, connectivity = 0, expansion_add = 0, expansion_search = 0, multi = false } = dimensionsOrConfigs);
        } else if ((typeof dimensionsOrConfigs === 'number' && !Number.isNaN(dimensionsOrConfigs)) || typeof dimensionsOrConfigs === 'bigint') {
            // Parameters are provided as individual arguments
            dimensions = dimensionsOrConfigs;
        } else {
            throw new Error("Invalid arguments. Expected either individual arguments or a single object argument.");
        }

        if (!Number.isInteger(dimensions) || !Number.isInteger(connectivity) || !Number.isInteger(expansion_add) || !Number.isInteger(expansion_search) || dimensions <= 0 || connectivity < 0 || expansion_add < 0 || expansion_search < 0) {
            throw new Error("`dimensions`, `connectivity`, `expansion_add`, and `expansion_search` must be non-negative integers, with `dimensions` being positive.");
        }

        if (typeof multi !== 'boolean') {
            throw new Error("`multi` must be a boolean value.");
        }

        if (!Object.values(MetricKind).includes(metric)) {
            throw new Error(`Invalid metric: ${metric}. It must be one of: ${Object.values(MetricKind).join(', ')}`);
        }

        if (!Object.values(ScalarKind).includes(quantization)) {
            throw new Error(`Invalid quantization: ${quantization}. It must be one of: ${Object.values(ScalarKind).join(', ')}`);
        }

        this._compiledIndex = new compiled.CompiledIndex(dimensions, metric, quantization, connectivity, expansion_add, expansion_search, multi);
    }

    /**
     * Add vectors to the index.
     * 
     * This method accepts vectors and their corresponding keys for indexing.
     * Each key should correspond to a vector. If a single key is provided,
     * it is broadcasted to match the number of provided vectors.
     * 
     * Vectors should be provided as a flat typed array representing a matrix
     * where each row is a vector to be indexed. The matrix should have a size
     * of n * d, where n is the number of vectors, and d is the dimensionality
     * of the vectors.
     * 
     * Keys should be provided as a BigInt or an array-like object of BigInts
     * representing the unique identifier for each vector.
     * 
     * @param {bigint|bigint[]|BigUint64Array} keys - Input identifiers for every vector.
     *        If a single key is provided, it is associated with all provided vectors.
     * @param {Float32Array|Float64Array|Int8Array} vectors - Input matrix representing vectors,
     *        matrix of size n * d, where n is the number of vectors, and d is their dimensionality.
     * @throws Will throw an error if the length of keys doesn't match the number of vectors
     *         or if it's not a single key.
     */
    add(keys, vectors) {
        let normalizedKeys = normalizeKeys(keys);
        let normalizedVectors = normalizeVectors(vectors, this._compiledIndex.dimensions());
        let countVectors = normalizedVectors.length / this._compiledIndex.dimensions();

        // If a single key is provided but there are multiple vectors,
        // broadcast the single key value to match the number of vectors
        if (normalizedKeys.length === 1 && countVectors > 1) {
            normalizedKeys = BigUint64Array.from({ length: countVectors }, () => normalizedKeys[0]);
        } else if (normalizedKeys.length !== countVectors) {
            throw new Error(`The length of keys (${normalizedKeys.length}) must match the number of vectors (${countVectors}) or be a single key.`);
        }

        // Call the compiled method
        this._compiledIndex.add(normalizedKeys, normalizedVectors);
    }

    /**
     * Perform a k-nearest neighbor search on the index.
     * 
     * This method accepts a matrix of query vectors and returns the closest vectors
     * from the index for each query. The method returns an object containing the keys,
     * distances, and counts of the matches found.
     * 
     * Vectors should be provided as a flat typed array representing a matrix where
     * each row is a vector. The matrix should be of size n * d, where n is the
     * number of query vectors, and d is their dimensionality.
     * 
     * The parameter `k` specifies the number of nearest neighbors to return for each
     * query vector. If there are not enough results for a query, the result array is
     * padded with -1s.
     * 
     * @param {Float32Array|Float64Array|Int8Array|Array<Array<number>>} vectors - Input matrix representing query vectors, can be a TypedArray or an array of arrays.
     * @param {number} k - The number of nearest neighbors to search for each query vector.
     * @return {Matches|BatchMatches} - Search results for one or more queries, containing keys, distances, and counts of the matches found.
     * @throws Will throw an error if `k` is not a positive integer or if the size of the vectors is not a multiple of dimensions.
     * @throws Will throw an error if `vectors` is not a valid input type (TypedArray or an array of arrays) or if its flattened size is not a multiple of dimensions.
     */
    search(vectors, k) {
        if ((!Number.isNaN(k) && typeof k !== 'number') || k <= 0) {
            throw new Error("`k` must be a positive integer representing the number of nearest neighbors to search for.");
        }

        const normalizedVectors = normalizeVectors(vectors, this._compiledIndex.dimensions());

        // Call the compiled method and create Matches or BatchMatches object with the result
        const result = this._compiledIndex.search(normalizedVectors, k);
        const countInQueries = normalizedVectors.length / Number(this._compiledIndex.dimensions());
        const batchMatches = new BatchMatches(result[0], result[1], result[2], k);

        if (countInQueries === 1) {
            return batchMatches.get(0);
        } else {
            return batchMatches;
        }
    }

    /**
     * Verifies the presence of one or more keys in the index.
     * 
     * This method accepts one or multiple keys as input and returns a boolean or 
     * an array of booleans indicating whether each key is present in the index.
     * 
     * @param {bigint|bigint[]|BigUint64Array} keys - The identifier(s) of the vector(s) to be checked for presence in the index.
     * @return {boolean|boolean[]} - Returns true if a single key is contained in the index, false otherwise. Returns an array of booleans corresponding to the presence of each key in the index when multiple keys are provided.
     * @throws Will throw an error if keys are not integers.
     */
    contains(keys) {
        let normalizedKeys = normalizeKeys(keys);
        let normalizedResults = this._compiledIndex.contains(normalizedKeys);
        if (isOneKey(keys))
            return normalizedResults[0];
        else
            return normalizedResults;
    }

    /**
     * Counts the number of times keys shows up in the index.
     * 
     * @param {bigint|bigint[]|BigUint64Array} keys - The identifier(s) of the vector(s) to be enumerated.
     * @return {number|number[]} - Returns the number of vectors found when a single key is provided. Returns an array of big integers corresponding to the number of vectors found for each key when multiple keys are provided.
     * @throws Will throw an error if keys are not integers.
     */
    count(keys) {
        let normalizedKeys = normalizeKeys(keys);
        let normalizedResults = this._compiledIndex.count(normalizedKeys);
        if (isOneKey(keys))
            return normalizedResults[0];
        else
            return normalizedResults;
    }

    /**
     * Removes one or multiple vectors from the index.
     * 
     * This method accepts one or multiple keys as input and removes the corresponding vectors from the index.
     * It returns the number of vectors actually removed for each key provided.
     * 
     * @param {bigint|bigint[]|BigUint64Array} keys - The identifier(s) of the vector(s) to be removed.
     * @return {number|number[]} - Returns the number of vectors deleted when a single key is provided. Returns an array of big integers corresponding to the number of vectors deleted for each key when multiple keys are provided.
     * @throws Will throw an error if keys are not integers.
     */
    remove(keys) {
        let normalizedKeys = normalizeKeys(keys);
        normalizedResults = this._compiledIndex.remove(normalizedKeys);
        if (isOneKey(keys))
            return normalizedResults[0];
        else
            return normalizedResults;
    }

    /**
     * Returns the dimensionality of vectors.
     * @return {number} The dimensionality of vectors.
     */
    dimensions() { return this._compiledIndex.dimensions() }

    /**
     * Returns connectivity.
     * @return {number} The connectivity of index.
     */
    connectivity() { return this._compiledIndex.connectivity() }

    /**
     * Returns the number of vectors currently indexed.
     * @return {number} The number of vectors currently indexed.
     */
    size() { return this._compiledIndex.size() }

    /**
     * Returns index capacity.
     * @return {number} The capacity of index.
     */
    capacity() { return this._compiledIndex.capacity() }

    /** 
     * Write index to a file.
     * @param {string} path File path to write.
     * @throws Will throw an error if `path` is not a string.
     */
    save(path) {
        if (typeof path !== 'string') throw new Error("`path` must be a string representing the file path to write.");
        this._compiledIndex.save(path);
    }

    /** 
     * Load index from a file.
     * @param {string} path File path to read.
     * @throws Will throw an error if `path` is not a string.
     */
    load(path) {
        if (typeof path !== 'string') throw new Error("`path` must be a string representing the file path to read.");
        this._compiledIndex.load(path);
    }

    /** 
     * View index from a file, without loading into RAM.
     * @param {string} path File path to read.
     * @throws Will throw an error if `path` is not a string.
     */
    view(path) {
        if (typeof path !== 'string') throw new Error("`path` must be a string representing the file path to read.");
        this._compiledIndex.view(path);
    }
}

/**
 * Performs an exact search on the given dataset to find the best matching vectors for each query.
 * 
 * @param {Float32Array|Float64Array|Int8Array|Array<Array<number>>} dataset - The dataset containing vectors to be searched. It can be a TypedArray or an array of arrays.
 * @param {Float32Array|Float64Array|Int8Array|Array<Array<number>>} queries - The queries containing vectors to search for in the dataset. It can be a TypedArray or an array of arrays.
 * @param {number} dimensions - The dimensionality of the vectors in both the dataset and the queries. It defines the number of elements in each vector.
 * @param {number} count - The number of nearest neighbors to return for each query. If the dataset contains fewer vectors than the specified count, the result will contain only the available vectors.
 * @param {MetricKind} metric - The distance metric to be used for the search.
 * @return {Matches|BatchMatches} - Returns a `Matches` or `BatchMatches` object containing the results of the search.
 * @throws Will throw an error if `dimensions` and `count` are not positive integers.
 * @throws Will throw an error if `metric` is not a valid MetricKind.
 * @throws Will throw an error if `dataset` and `queries` are not valid input types (TypedArray or an array of arrays).
 * @throws Will throw an error if the sizes of the flattened `dataset` and `queries` are not multiples of `dimensions`.
 * @throws Will throw an error if `count` is greater than the number of vectors in the `dataset`.
 * 
 * @example
 * const dataset = [[1.0, 2.0], [3.0, 4.0]]; // Two vectors: [1.0, 2.0] and [3.0, 4.0]
 * const queries = [[1.5, 2.5]]; // One vector: [1.5, 2.5]
 * const dimensions = 2; // The number of elements in each vector.
 * const count = 1; // The number of nearest neighbors to return for each query.
 * const metric = MetricKind.IP; // Using the Inner Product distance metric.
 * 
 * const result = exactSearch(dataset, queries, dimensions, count, metric);
 * // result might be: 
 * // {
 * //    keys: BigUint64Array [ 1n ],
 * //    distances: Float32Array [ some_value ],
 * // }
 */
function exactSearch(dataset, queries, dimensions, count, metric) {

    // Validate and normalize the dimensions and count
    dimensions = Number(dimensions);
    count = Number(count);
    if (count <= 0 || dimensions <= 0) {
        throw new Error("Dimensions and count must be positive integers.");
    }

    // Validate metric
    if (!Object.values(MetricKind).includes(metric)) {
        throw new Error(`Invalid metric: ${metric}. It must be one of: ${Object.values(MetricKind).join(', ')}`);
    }

    // Flatten and normalize dataset and queries if they are arrays of arrays
    let targetType;
    if (dataset instanceof Float64Array) targetType = Float64Array;
    else if (dataset instanceof Int8Array) targetType = Int8Array;
    else targetType = Float32Array; // default to Float32Array if dataset is not Float64Array or Int8Array

    dataset = normalizeVectors(dataset, dimensions, targetType);
    queries = normalizeVectors(queries, dimensions, targetType);
    const countInDataset = dataset.length / dimensions;
    const countInQueries = queries.length / dimensions;
    if (count > countInDataset) {
        throw new Error("Count must be equal or smaller than the number of vectors in the dataset.");
    }

    // Call the compiled function with the normalized input
    const result = compiled.exactSearch(dataset, queries, dimensions, count, metric);

    // Create and return a Matches or BatchMatches object with the result
    if (countInQueries == 1) {
        return new Matches(result[0], result[1]);
    } else {
        return new BatchMatches(result[0], result[1], result[2], count);
    }
}

module.exports = {
    Index,
    MetricKind,
    ScalarKind,
    Matches,
    BatchMatches,
    exactSearch,
};
