using System;
using System.Runtime.InteropServices;
using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

/// <summary>
/// USearchIndex class provides a managed wrapper for the USearch library's index functionality.
/// </summary>
public class USearchIndex : IDisposable
{
    private IntPtr _index;
    private bool _disposedValue = false;
    private ulong _cachedDimensions;

    /// <summary>
    /// Initializes a new instance of the USearchIndex class with specified options.
    /// </summary>
    /// <param name="metricKind">The metric kind used for distance calculation between vectors.</param>
    /// <param name="quantization">The scalar kind used for quantization of vector data during indexing.</param>
    /// <param name="dimensions">The number of dimensions in the vectors to be indexed.</param>
    /// <param name="connectivity">The optional connectivity parameter that limits connections-per-node in the graph.</param>
    /// <param name="expansionAdd">The optional expansion factor used for index construction when adding vectors.</param>
    /// <param name="expansionSearch">The optional expansion factor used for index construction during search operations.</param>
    /// <param name="multi">When set allows multiple vectors to map to the same key.</param>
    public USearchIndex(
        MetricKind metricKind,
        ScalarKind quantization,
        ulong dimensions,
        ulong connectivity = 0,
        ulong expansionAdd = 0,
        ulong expansionSearch = 0,
        bool multi = false
    )
    {
        IndexOptions initOptions = new()
        {
            metric_kind = metricKind,
            metric = default,
            quantization = quantization,
            dimensions = dimensions,
            connectivity = connectivity,
            expansion_add = expansionAdd,
            expansion_search = expansionSearch,
            multi = multi
        };

        this._index = usearch_init(ref initOptions, out IntPtr error);
        HandleError(error);
        this._cachedDimensions = dimensions;
    }

    /// <summary>
    /// Initializes a new instance of the USearchIndex class with specified options.
    /// </summary>
    /// <param name="options">The options structure containing initialization parameters.</param>
    public USearchIndex(IndexOptions options)
    {
        this._index = usearch_init(ref options, out IntPtr error);
        HandleError(error);
        this._cachedDimensions = options.dimensions;
    }

    /// <summary>
    /// Initializes a new instance of the USearchIndex class and loads or views the index from a specified file.
    /// </summary>
    /// <param name="path">The file path from where the index will be loaded or viewed.</param>
    /// <param name="view">If true, creates a view of the index without copying it into memory.</param>
    public USearchIndex(string path, bool view = false)
    {
        IndexOptions initOptions = new();
        this._index = usearch_init(ref initOptions, out IntPtr error);
        HandleError(error);

        if (view)
        {
            usearch_view(this._index, path, out error);
        }
        else
        {
            usearch_load(this._index, path, out error);
        }

        HandleError(error);

        this._cachedDimensions = this.Dimensions();
    }

    /// <summary>
    /// Saves the index to a specified file.
    /// </summary>
    /// <param name="path">The file path where the index will be saved.</param>
    public void Save(string path)
    {
        usearch_save(this._index, path, out IntPtr error);
        HandleError(error);
    }

    /// <summary>
    /// Gets the current size (number of vectors) of the index.
    /// </summary>
    /// <returns>The number of vectors in the index.</returns>
    public ulong Size()
    {
        ulong size = (ulong)usearch_size(this._index, out IntPtr error);
        HandleError(error);
        return size;
    }

    /// <summary>
    /// Gets the current capacity (number of vectors) of the index.
    /// </summary>
    /// <returns>The capacity of the index.</returns>
    public ulong Capacity()
    {
        ulong capacity = (ulong)usearch_capacity(this._index, out IntPtr error);
        HandleError(error);
        return capacity;
    }

    /// <summary>
    /// Gets the number of dimensions in the vectors in the index.
    /// </summary>
    /// <returns>The number of dimensions.</returns>
    public ulong Dimensions()
    {
        ulong dimensions = (ulong)usearch_dimensions(this._index, out IntPtr error);
        HandleError(error);
        return dimensions;
    }

    /// <summary>
    /// Gets the connectivity parameter of the index.
    /// </summary>
    /// <returns>The connectivity parameter.</returns>
    public ulong Connectivity()
    {
        ulong connectivity = (ulong)usearch_connectivity(this._index, out IntPtr error);
        HandleError(error);
        return connectivity;
    }

    /// <summary>
    /// Checks if the index contains a vector with a specific key.
    /// </summary>
    /// <param name="key">The key to be checked.</param>
    /// <returns>True if the index contains the vector with the given key, false otherwise.</returns>
    public bool Contains(ulong key)
    {
        bool result = usearch_contains(this._index, key, out IntPtr error);
        HandleError(error);
        return result;
    }

    /// <summary>
    /// Counts the number of entries in the index under a specific key.
    /// </summary>
    /// <param name="key">The key to be checked.</param>
    /// <returns>The number of vectors found under that key.</returns>
    public int Count(ulong key)
    {
        int count = checked((int)usearch_count(this._index, key, out IntPtr error));
        HandleError(error);
        return count;
    }

    /// <summary>
    /// Reserves additional capacity in the index.
    /// </summary>
    /// <param name="size">The number of new vectors to reserve capacity for.</param>
    private void IncreaseCapacity(ulong size)
    {
        usearch_reserve(this._index, (UIntPtr)(this.Size() + size), out IntPtr error);
        HandleError(error);
    }

    /// <summary>
    /// Checks if the index has enough capacity to add a specific number of vectors.
    /// If not, increases the capacity.
    /// </summary>
    /// <param name="size_increase">The number of vectors to be added.</param>
    private void CheckIncreaseCapacity(ulong size_increase)
    {
        ulong size_demand = this.Size() + size_increase;
        if (this.Capacity() < size_demand)
        {
            this.IncreaseCapacity(size_increase);
        }
    }

    /// <summary>
    /// Adds a vector with a specific key to the index.
    /// </summary>
    /// <param name="key">The key associated with the vector.</param>
    /// <param name="vector">The vector data to be added.</param>
    public void Add(ulong key, float[] vector)
    {
        this.CheckIncreaseCapacity(1);
        usearch_add(this._index, key, vector, ScalarKind.Float32, out IntPtr error);
        HandleError(error);
    }

    /// <summary>
    /// Adds a vector with a specific key to the index.
    /// </summary>
    /// <param name="key">The key associated with the vector.</param>
    /// <param name="vector">The vector data to be added.</param>
    public void Add(ulong key, double[] vector)
    {
        this.CheckIncreaseCapacity(1);
        usearch_add(this._index, key, vector, ScalarKind.Float64, out IntPtr error);
        HandleError(error);
    }

    /// <summary>
    /// Adds multiple vectors with specific keys to the index.
    /// </summary>
    /// <param name="keys">The keys associated with the vectors.</param>
    /// <param name="vectors">The vector data to be added.</param>
    public void Add(ulong[] keys, float[][] vectors)
    {
        this.CheckIncreaseCapacity((ulong)vectors.Length);
        for (int i = 0; i < vectors.Length; i++)
        {
            usearch_add(this._index, keys[i], vectors[i], ScalarKind.Float32, out IntPtr error);
            HandleError(error);
        }
    }

    /// <summary>
    /// Adds multiple vectors with specific keys to the index.
    /// </summary>
    /// <param name="keys">The keys associated with the vectors.</param>
    /// <param name="vectors">The vector data to be added.</param>
    public void Add(ulong[] keys, double[][] vectors)
    {
        this.CheckIncreaseCapacity((ulong)vectors.Length);
        for (int i = 0; i < vectors.Length; i++)
        {
            usearch_add(this._index, keys[i], vectors[i], ScalarKind.Float64, out IntPtr error);
            HandleError(error);
        }
    }

    /// <summary>
    /// Retrieves the vector associated with the given key from the index.
    /// </summary>
    /// <param name="key">The key of the vector to retrieve.</param>
    /// <param name="vector">The vector data retrieved from the index.</param>
    /// <returns>The number of vectors found under that key.</returns>
    public int Get(ulong key, out float[] vector)
    {
        vector = new float[this._cachedDimensions];
        int foundVectorsCount = checked((int)usearch_get(this._index, key, (UIntPtr)1, vector, ScalarKind.Float32, out IntPtr error));
        HandleError(error);
        if (foundVectorsCount < 1)
        {
            vector = null;
        }

        return foundVectorsCount;
    }

    /// <summary>
    /// Retrieves multiple vectors associated with the given key from the index.
    /// </summary>
    /// <param name="key">The key of the vectors to retrieve.</param>
    /// <param name="count">The number of vectors to retrieve.</param>
    /// <param name="vectors">The vectors data retrieved from the index.</param>
    /// <returns>The number of vectors found under that key.</returns>
    public int Get(ulong key, int count, out float[][] vectors)
    {
        var flattenVectors = new float[count * (int)this._cachedDimensions];
        int foundVectorsCount = checked((int)usearch_get(this._index, key, (UIntPtr)count, flattenVectors, ScalarKind.Float32, out IntPtr error));
        HandleError(error);
        if (foundVectorsCount < 1)
        {
            vectors = null;
        }
        else
        {
            vectors = new float[foundVectorsCount][];
            for (int i = 0; i < foundVectorsCount; i++)
            {
                vectors[i] = new float[this._cachedDimensions];
                Array.Copy(flattenVectors, i * (int)this._cachedDimensions, vectors[i], 0, (int)this._cachedDimensions);
            }
        }

        return foundVectorsCount;
    }

    /// <summary>
    /// Retrieves the vector associated with the given key from the index.
    /// </summary>
    /// <param name="key">The key of the vector to retrieve.</param>
    /// <param name="vector">The vector data retrieved from the index.</param>
    /// <returns>The number of vectors found under that key.</returns>
    public int Get(ulong key, out double[] vector)
    {
        vector = new double[this._cachedDimensions];
        int foundVectorsCount = checked((int)usearch_get(this._index, key, (UIntPtr)1, vector, ScalarKind.Float64, out IntPtr error));
        HandleError(error);
        if (foundVectorsCount < 1)
        {
            vector = null;
        }

        return foundVectorsCount;
    }

    /// <summary>
    /// Retrieves multiple vectors associated with the given key from the index.
    /// </summary>
    /// <param name="key">The key of the vectors to retrieve.</param>
    /// <param name="count">The number of vectors to retrieve.</param>
    /// <param name="vectors">The vectors data retrieved from the index.</param>
    /// <returns>The number of vectors found under that key.</returns>
    public int Get(ulong key, int count, out double[][] vectors)
    {
        var flattenVectors = new double[count * (int)this._cachedDimensions];
        int foundVectorsCount = checked((int)usearch_get(this._index, key, (UIntPtr)count, flattenVectors, ScalarKind.Float64, out IntPtr error));
        HandleError(error);
        if (foundVectorsCount < 1)
        {
            vectors = null;
        }
        else
        {
            vectors = new double[foundVectorsCount][];
            for (int i = 0; i < foundVectorsCount; i++)
            {
                vectors[i] = new double[this._cachedDimensions];
                Array.Copy(flattenVectors, i * (int)this._cachedDimensions, vectors[i], 0, (int)this._cachedDimensions);
            }
        }

        return foundVectorsCount;
    }

    /// <summary>
    /// Searches for the closest vectors to the query vector.
    /// </summary>
    /// <param name="queryVector">The query vector data.</param>
    /// <param name="count">The number of nearest neighbors to search.</param>
    /// <param name="keys">The keys of the nearest neighbors found.</param>
    /// <param name="distances">The distances to the nearest neighbors found.</param>
    /// <param name="scalarKind">The scalar type used in the query vector data.</param>
    /// <returns>The number of matches found.</returns>
    private int Search<T>(T[] queryVector, int count, out ulong[] keys, out float[] distances, ScalarKind scalarKind)
    {
        keys = new ulong[count];
        distances = new float[count];

        GCHandle handle = GCHandle.Alloc(queryVector, GCHandleType.Pinned);
        int matches = 0;
        try
        {
            IntPtr queryVectorPtr = handle.AddrOfPinnedObject();
            matches = checked((int)usearch_search(this._index, queryVectorPtr, scalarKind, (UIntPtr)count, keys, distances, out IntPtr error));
            HandleError(error);
        }
        finally
        {
            handle.Free();
        }

        if (matches < count)
        {
            Array.Resize(ref keys, (int)matches);
            Array.Resize(ref distances, (int)matches);
        }

        return matches;
    }

    /// <summary>
    /// Searches for the closest vectors to the query vector.
    /// </summary>
    /// <param name="queryVector">The query vector data.</param>
    /// <param name="count">The number of nearest neighbors to search.</param>
    /// <param name="keys">The keys of the nearest neighbors found.</param>
    /// <param name="distances">The distances to the nearest neighbors found.</param>
    /// <returns>The number of matches found.</returns>
    public int Search(float[] queryVector, int count, out ulong[] keys, out float[] distances)
    {
        return this.Search(queryVector, count, out keys, out distances, ScalarKind.Float32);
    }

    /// <summary>
    /// Searches for the closest vectors to the query vector.
    /// </summary>
    /// <param name="queryVector">The query vector data.</param>
    /// <param name="count">The number of nearest neighbors to search.</param>
    /// <param name="keys">The keys of the nearest neighbors found.</param>
    /// <param name="distances">The distances to the nearest neighbors found.</param>
    /// <returns>The number of matches found.</returns>
    public int Search(double[] queryVector, int count, out ulong[] keys, out float[] distances)
    {
        return this.Search(queryVector, count, out keys, out distances, ScalarKind.Float64);
    }

    /// <summary>
    /// Removes the vector associated with the given key from the index.
    /// </summary>
    /// <param name="key">The key of the vector to be removed.</param>
    /// <returns>The number of vectors removed.</returns>
    public int Remove(ulong key)
    {
        int removedCount = checked((int)usearch_remove(this._index, key, out IntPtr error));
        HandleError(error);
        return removedCount;
    }

    /// <summary>
    /// Renames the vector to map to a different key.
    /// </summary>
    /// <param name="keyFrom">The current key of the vector.</param>
    /// <param name="keyTo">The new key for the vector.</param>
    /// <returns>The number of vectors renamed.</returns>
    public int Rename(ulong keyFrom, ulong keyTo)
    {
        int foundVectorsCount = checked((int)usearch_rename(this._index, keyFrom, keyTo, out IntPtr error));
        HandleError(error);
        return foundVectorsCount;
    }

    /// <summary>
    /// Handles errors by throwing a USearchException if the error pointer is not null.
    /// </summary>
    /// <param name="error">The error pointer returned by USearch functions.</param>
    private static void HandleError(IntPtr error)
    {
        if (error != IntPtr.Zero)
        {
            throw new USearchException($"USearch operation failed: {Marshal.PtrToStringAnsi(error)}");
        }
    }

    /// <summary>
    /// Frees the resources associated with the index.
    /// </summary>
    private void FreeIndex()
    {
        if (this._index != IntPtr.Zero)
        {
            usearch_free(this._index, out IntPtr error);
            HandleError(error);
            this._index = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
    /// </summary>
    public void Dispose()
    {
        this.Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases the unmanaged resources used by the USearchIndex and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">If true, release both managed and unmanaged resources; otherwise, release only unmanaged resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!this._disposedValue)
        {
            this.FreeIndex();
            this._disposedValue = true;
        }
    }


    /// <summary>
    /// Destructor for the USearchIndex class.
    /// </summary>
    ~USearchIndex() => this.Dispose(false);
}
