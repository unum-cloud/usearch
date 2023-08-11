

using System;
using System.Runtime.InteropServices;

public enum usearch_metric_kind_t : uint
{
    usearch_metric_unknown_k = 0,
    usearch_metric_cos_k,
    usearch_metric_ip_k,
    usearch_metric_l2sq_k,
    usearch_metric_haversine_k,
    usearch_metric_pearson_k,
    usearch_metric_jaccard_k,
    usearch_metric_hamming_k,
    usearch_metric_tanimoto_k,
    usearch_metric_sorensen_k,
}

public enum usearch_scalar_kind_t : uint
{
    usearch_scalar_unknown_k = 0,
    usearch_scalar_f32_k,
    usearch_scalar_f64_k,
    usearch_scalar_f16_k,
    usearch_scalar_i8_k,
    usearch_scalar_b1_k,
}

// TODO implement custom metric delegate following microsoft guides:
// 1) https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices
// 2) https://learn.microsoft.com/en-us/dotnet/framework/interop/marshalling-a-delegate-as-a-callback-method
// public delegate float usearch_metric_t(IntPtr a, IntPtr b);

[StructLayout(LayoutKind.Sequential)]
public struct usearch_init_options_t
{
    public usearch_metric_kind_t metric_kind;
    public IntPtr metric;
    public usearch_scalar_kind_t quantization;
    public UIntPtr dimensions;
    public UIntPtr connectivity;
    public UIntPtr expansion_add;
    public UIntPtr expansion_search;

    public usearch_init_options_t(
        usearch_metric_kind_t metric_kind = usearch_metric_kind_t.usearch_metric_unknown_k,
        IntPtr metric = default,
        usearch_scalar_kind_t quantization = usearch_scalar_kind_t.usearch_scalar_unknown_k,
        UIntPtr dimensions = default,
        UIntPtr connectivity = default,
        UIntPtr expansion_add = default,
        UIntPtr expansion_search = default
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

