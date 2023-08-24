using System;
using System.Runtime.InteropServices;

namespace USearchNET
{
    public enum MetricKind : uint
    {
        Unknown = 0,
        Cos,
        Ip,
        L2sq,
        Haversine,
        Pearson,
        Jaccard,
        Hamming,
        Tanimoto,
        Sorensen,
    }

    public enum ScalarKind : uint
    {
        Unknown = 0,
        Float32,
        Float64,
        Float16,
        Int8,
        Byte1,
    }

    // TODO implement custom metric delegate following microsoft guides:
    // 1) https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices
    // 2) https://learn.microsoft.com/en-us/dotnet/framework/interop/marshalling-a-delegate-as-a-callback-method
    // public delegate float usearch_metric_t(IntPtr a, IntPtr b);

    [StructLayout(LayoutKind.Sequential)]
    public struct IndexOptions
    {
        public MetricKind metric_kind;
        public IntPtr metric;
        public ScalarKind quantization;
        public ulong dimensions;
        public ulong connectivity;
        public ulong expansion_add;
        public ulong expansion_search;

        public IndexOptions(
            MetricKind metric_kind = MetricKind.Unknown,
            IntPtr metric = default,
            ScalarKind quantization = ScalarKind.Unknown,
            ulong dimensions = 0,
            ulong connectivity = 0,
            ulong expansion_add = 0,
            ulong expansion_search = 0
        )
        {
            this.metric_kind = metric_kind;
            this.metric = metric;
            this.quantization = quantization;
            this.dimensions = dimensions;
            this.connectivity = connectivity;
            this.expansion_add = expansion_add;
            this.expansion_search = expansion_search;
        }
    }

}
