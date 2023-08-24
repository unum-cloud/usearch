using System;
using System.Runtime.InteropServices;
using static USearchNET.NativeMethods;

namespace USearchNET
{

    public class USearchIndex : IDisposable
    {
        private IntPtr _index;
        private bool _disposedValue = false;
        private ulong _cachedDimensions;

        public USearchIndex(
            MetricKind metricKind,
            ScalarKind quantization,
            ulong dimensions,
            ulong connectivity = 0,
            ulong expansionAdd = 0,
            ulong expansionSearch = 0
        // CustomDistanceFunction? customMetric = null
        )
        {
            IndexOptions initOptions = new IndexOptions
            {
                metric_kind = metricKind,
                metric = default,
                quantization = quantization,
                dimensions = dimensions,
                connectivity = connectivity,
                expansion_add = expansionAdd,
                expansion_search = expansionSearch
            };

            this._index = usearch_init(ref initOptions, out IntPtr error);
            HandleError(error);
            this._cachedDimensions = dimensions;
        }

        public USearchIndex(IndexOptions options)
        {
            IndexOptions initOptions = options;
            this._index = usearch_init(ref initOptions, out IntPtr error);
            HandleError(error);
            this._cachedDimensions = options.dimensions;
        }

        public USearchIndex(string path, bool view = false)
        {
            IndexOptions initOptions = new IndexOptions();
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

        public void Save(string path)
        {
            usearch_save(this._index, path, out IntPtr error);
            HandleError(error);
        }

        public ulong Size()
        {
            ulong size = (ulong)usearch_size(this._index, out IntPtr error);
            HandleError(error);
            return size;
        }

        public ulong Capacity()
        {
            ulong capacity = (ulong)usearch_capacity(this._index, out IntPtr error);
            HandleError(error);
            return capacity;
        }

        public ulong Dimensions()
        {
            ulong dimensions = (ulong)usearch_dimensions(this._index, out IntPtr error);
            HandleError(error);
            return dimensions;
        }

        public ulong Connectivity()
        {
            ulong connectivity = (ulong)usearch_connectivity(this._index, out IntPtr error);
            HandleError(error);
            return connectivity;
        }

        public bool Contains(ulong key)
        {
            bool result = usearch_contains(this._index, key, out IntPtr error);
            HandleError(error);
            return result;
        }

        private void IncreaseCapacity(ulong size)
        {
            usearch_reserve(this._index, (UIntPtr)(this.Size() + size), out IntPtr error);
            HandleError(error);
        }

        private void CheckIncreaseCapacity(ulong size_increase)
        {
            ulong size_demand = this.Size() + size_increase;
            if (this.Capacity() < size_demand)
            {
                this.IncreaseCapacity(size_increase);
            }
        }

        public void Add(ulong key, float[] vector)
        {
            this.CheckIncreaseCapacity(1);
            usearch_add(this._index, key, vector, ScalarKind.Float32, out IntPtr error);
            HandleError(error);
        }

        public void Add(ulong key, double[] vector)
        {
            this.CheckIncreaseCapacity(1);
            usearch_add(this._index, key, vector, ScalarKind.Float64, out IntPtr error);
            HandleError(error);
        }

        public void Add(ulong[] keys, float[][] vectors)
        {
            this.CheckIncreaseCapacity((ulong)vectors.Length);
            for (int i = 0; i < vectors.Length; i++)
            {
                usearch_add(this._index, keys[i], vectors[i], ScalarKind.Float32, out IntPtr error);
                HandleError(error);
            }
        }

        public void Add(ulong[] keys, double[][] vectors)
        {
            this.CheckIncreaseCapacity((ulong)vectors.Length);
            for (int i = 0; i < vectors.Length; i++)
            {
                usearch_add(this._index, keys[i], vectors[i], ScalarKind.Float64, out IntPtr error);
                HandleError(error);
            }
        }

        public bool Get(ulong key, out float[] vector)
        {
            vector = new float[this._cachedDimensions];
            bool success = usearch_get(this._index, key, vector, ScalarKind.Float32, out IntPtr error);
            HandleError(error);
            if (!success)
            {
                vector = null;
            }

            return success;
        }

        public bool Get(ulong key, out double[] vector)
        {
            vector = new double[this._cachedDimensions];
            bool success = usearch_get(this._index, key, vector, ScalarKind.Float64, out IntPtr error);
            HandleError(error);
            if (!success)
            {
                vector = null;
            }

            return success;
        }

        private int Search<T>(T[] queryVector, int resultsLimit, out ulong[] keys, out float[] distances, ScalarKind scalarKind)
        {
            keys = new ulong[resultsLimit];
            distances = new float[resultsLimit];

            GCHandle handle = GCHandle.Alloc(queryVector, GCHandleType.Pinned);
            int matches = 0;
            try
            {
                IntPtr queryVectorPtr = handle.AddrOfPinnedObject();
                matches = checked((int)usearch_search(this._index, queryVectorPtr, scalarKind, (UIntPtr)resultsLimit, keys, distances, out IntPtr error));
                HandleError(error);
            }
            finally
            {
                handle.Free();
            }

            if (matches < resultsLimit)
            {
                Array.Resize(ref keys, (int)matches);
                Array.Resize(ref distances, (int)matches);
            }

            return matches;
        }

        public int Search(float[] queryVector, int resultsLimit, out ulong[] keys, out float[] distances)
        {
            return this.Search(queryVector, resultsLimit, out keys, out distances, ScalarKind.Float32);
        }

        public int Search(double[] queryVector, int resultsLimit, out ulong[] keys, out float[] distances)
        {
            return this.Search(queryVector, resultsLimit, out keys, out distances, ScalarKind.Float64);
        }

        public bool Remove(ulong key)
        {
            bool success = usearch_remove(this._index, key, out IntPtr error);
            HandleError(error);
            return success;
        }

        private static void HandleError(IntPtr error)
        {
            if (error != IntPtr.Zero)
            {
                Console.WriteLine($"Error {error}");
                throw new USearchException($"USearch operation failed: {Marshal.PtrToStringAnsi(error)}");
            }
        }

        private void FreeIndex()
        {
            if (this._index != IntPtr.Zero)
            {
                usearch_free(this._index, out IntPtr error);
                HandleError(error);
                this._index = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!this._disposedValue)
            {
                this.FreeIndex();
                this._disposedValue = true;
            }
        }

        ~USearchIndex() => this.Dispose(false);

        public class USearchException : Exception { public USearchException(string message) : base(message) { } }
    }
}
