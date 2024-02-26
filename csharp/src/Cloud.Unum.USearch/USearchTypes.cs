using System;
using System.Runtime.InteropServices;

namespace Cloud.Unum.USearch;

public enum MetricKind : uint
{
    Unknown = 0,
    Cos = 1,
    Ip = 2,
    L2sq = 3,
    Haversine = 4,
    Divergence = 5,
    Pearson = 6,
    Jaccard = 7,
    Hamming = 8,
    Tanimoto = 9,
    Sorensen = 10,
}

public enum ScalarKind : uint
{
    Unknown = 0,
    Float32 = 1,
    Float64 = 2,
    Float16 = 3,
    Int8 = 4,
    Bits1 = 5,
}

// TODO: implement custom metric delegate
// Microsoft guides links:
// 1) https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices
// 2) https://learn.microsoft.com/en-us/dotnet/framework/interop/marshalling-a-delegate-as-a-callback-method
// public delegate float CustomMetricFunction(IntPtr a, IntPtr b);

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

    [MarshalAs(UnmanagedType.Bool)]
    public bool multi;

    public IndexOptions(
        MetricKind metricKind = MetricKind.Unknown,
        IntPtr metric = default,
        ScalarKind quantization = ScalarKind.Unknown,
        ulong dimensions = 0,
        ulong connectivity = 0,
        ulong expansionAdd = 0,
        ulong expansionSearch = 0,
        bool multi = false
    )
    {
        this.metric_kind = metricKind;
        this.metric = default; // TODO: Use actual metric param, when support is added for custom metric delegate
        this.quantization = quantization;
        this.dimensions = dimensions;
        this.connectivity = connectivity;
        this.expansion_add = expansionAdd;
        this.expansion_search = expansionSearch;
        this.multi = multi;
    }
}

