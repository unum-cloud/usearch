
using System.Text;
using static NativeMethods;

public class USearchIndex : IDisposable
{
    private nint _index;
    private bool _disposedValue = false;
    private usearch_init_options_t _metaData;

    public USearchIndex(
        usearch_metric_kind_t metricKind,
        usearch_scalar_kind_t quantization,
        ulong dimensions,
        ulong connectivity = 0,
        ulong expansionAdd = 0,
        ulong expansionSearch = 0
    // CustomDistanceFunction? customMetric = null
    )
    {
        usearch_init_options_t initOptions = new usearch_init_options_t
        {
            metric_kind = metricKind,
            metric = null,
            quantization = quantization,
            dimensions = (nuint)dimensions,
            connectivity = (nuint)connectivity,
            expansion_add = (nuint)expansionAdd,
            expansion_search = (nuint)expansionSearch
        };

        _index = usearch_init(ref initOptions, out nint error);
        HandleError(error);
        _metaData = initOptions;
    }

    public USearchIndex(usearch_init_options_t options)
    {
        usearch_init_options_t initOptions = options;
        _index = usearch_init(ref initOptions, out nint error);
        HandleError(error);
        _metaData = initOptions;
    }

    public USearchIndex(string path, bool view = false)
    {
        usearch_init_options_t initOptions = new usearch_init_options_t();
        _index = usearch_init(ref initOptions, out nint error);
        HandleError(error);

        if (view)
            usearch_view(_index, path, out error);
        else
            usearch_load(_index, path, out error);

        _metaData = new usearch_init_options_t(
            dimensions: (nuint)Dimensions(),
            connectivity: (nuint)Connectivity()
        );

        HandleError(error);
    }

    public void Save(string path)
    {
        usearch_save(_index, path, out nint error);
        HandleError(error);
    }

    public ulong Size()
    {
        nuint size = usearch_size(_index, out nint error);
        HandleError(error);
        return size;
    }

    public usearch_init_options_t GetMetadata() => _metaData;

    public ulong Capacity()
    {
        nuint capacity = usearch_capacity(_index, out nint error);
        HandleError(error);
        return capacity;
    }

    public ulong Dimensions()
    {
        nuint dimensions = usearch_dimensions(_index, out nint error);
        HandleError(error);
        return dimensions;
    }

    public ulong Connectivity()
    {
        nuint connectivity = usearch_connectivity(_index, out nint error);
        HandleError(error);
        return connectivity;
    }

    public bool Contains(ulong key)
    {
        bool result = usearch_contains(_index, key, out nint error);
        HandleError(error);
        return result;
    }

    private void IncreaseCapacity(nuint size)
    {
        usearch_reserve(_index, (nuint)Size() + size, out nint error);
        HandleError(error);
    }

    private void CheckIncreaseCapacity(nuint size_increase)
    {
        nuint size_demand = (nuint)Size() + size_increase;
        if (Capacity() < size_demand)
            IncreaseCapacity(size_increase);
    }

    public void Add(ulong key, Half[] vector)
    {
        CheckIncreaseCapacity(1);
        usearch_add(_index, key, vector, usearch_scalar_kind_t.usearch_scalar_f16_k, out nint error);
        HandleError(error);
    }

    public void Add(ulong key, float[] vector)
    {
        CheckIncreaseCapacity(1);
        usearch_add(_index, key, vector, usearch_scalar_kind_t.usearch_scalar_f32_k, out nint error);
        HandleError(error);
    }

    public void Add(ulong key, double[] vector)
    {
        CheckIncreaseCapacity(1);
        usearch_add(_index, key, vector, usearch_scalar_kind_t.usearch_scalar_f64_k, out nint error);
        HandleError(error);
    }

    public void AddMany(ulong[] keys, Half[][] vectors)
    {
        CheckIncreaseCapacity((nuint)vectors.Length);
        for (int i = 0; i < vectors.Length; i++)
        {
            usearch_add(_index, keys[i], vectors[i], usearch_scalar_kind_t.usearch_scalar_f16_k, out nint error);
            HandleError(error);
        }
    }

    public void AddMany(ulong[] keys, float[][] vectors)
    {
        CheckIncreaseCapacity((nuint)vectors.Length);
        for (int i = 0; i < vectors.Length; i++)
        {
            usearch_add(_index, keys[i], vectors[i], usearch_scalar_kind_t.usearch_scalar_f32_k, out nint error);
            HandleError(error);
        }
    }

    public void AddMany(ulong[] keys, double[][] vectors)
    {
        CheckIncreaseCapacity((nuint)vectors.Length);
        for (int i = 0; i < vectors.Length; i++)
        {
            usearch_add(_index, keys[i], vectors[i], usearch_scalar_kind_t.usearch_scalar_f64_k, out nint error);
            HandleError(error);
        }
    }

    public bool Get(ulong key, out Half[]? vector)
    {
        vector = new Half[_metaData.dimensions];
        bool success = usearch_get(_index, key, vector, usearch_scalar_kind_t.usearch_scalar_f16_k, out nint error);
        HandleError(error);
        if (!success)
            vector = null;
        return success;
    }

    public bool Get(ulong key, out float[]? vector)
    {
        vector = new float[_metaData.dimensions];
        bool success = usearch_get(_index, key, vector, usearch_scalar_kind_t.usearch_scalar_f32_k, out nint error);
        HandleError(error);
        if (!success)
            vector = null;
        return success;
    }

    public bool Get(ulong key, out double[]? vector)
    {
        vector = new double[_metaData.dimensions];
        bool success = usearch_get(_index, key, vector, usearch_scalar_kind_t.usearch_scalar_f64_k, out nint error);
        HandleError(error);
        if (!success)
            vector = null;
        return success;
    }

    private ulong Search<T>(T[] queryVector, ulong resultsLimit, out Dictionary<ulong, float> foundKeyDistances, usearch_scalar_kind_t scalarKind)
    {
        ulong[] keys = new ulong[resultsLimit];
        float[] distances = new float[resultsLimit];

        GCHandle handle = GCHandle.Alloc(queryVector, GCHandleType.Pinned);
        ulong matches = 0;
        try
        {
            IntPtr queryVectorPtr = handle.AddrOfPinnedObject();
            matches = usearch_search(_index, queryVectorPtr, scalarKind, (nuint)resultsLimit, keys, distances, out nint error);
            HandleError(error);
        }
        finally
        {
            handle.Free();
        }
        foundKeyDistances = new Dictionary<ulong, float>();
        for (ulong i = 0; i < matches; i++)
            foundKeyDistances.Add(keys[i], distances[i]);
        return matches;
    }

    public ulong Search(Half[] queryVector, ulong resultsLimit, out Dictionary<ulong, float> foundKeyDistances)
    {
        return Search(queryVector, resultsLimit, out foundKeyDistances, usearch_scalar_kind_t.usearch_scalar_f16_k);
    }

    public ulong Search(float[] queryVector, ulong resultsLimit, out Dictionary<ulong, float> foundKeyDistances)
    {
        return Search(queryVector, resultsLimit, out foundKeyDistances, usearch_scalar_kind_t.usearch_scalar_f32_k);
    }

    public ulong Search(double[] queryVector, ulong resultsLimit, out Dictionary<ulong, float> foundKeyDistances)
    {
        return Search(queryVector, resultsLimit, out foundKeyDistances, usearch_scalar_kind_t.usearch_scalar_f64_k);
    }

    public bool Remove(ulong key)
    {
        bool success = usearch_remove(_index, key, out nint error);
        HandleError(error);
        return success;
    }

    private static void HandleError(nint error)
    {
        if (error != 0)
        {
            Console.WriteLine($"Error {error}");
            throw new USearchException($"USearch operation failed: {Marshal.PtrToStringAnsi(error)}");
        }
    }

    private void FreeIndex()
    {
        if (_index == 0)
            return;

        usearch_free(_index, out nint error);
        HandleError(error);
        _index = 0;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            FreeIndex();
            _disposedValue = true;
        }
    }

    ~USearchIndex() => Dispose(false);

    public class USearchException : Exception { public USearchException(string message) : base(message) { } }
}
