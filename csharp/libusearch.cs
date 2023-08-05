using System.Runtime.InteropServices;

namespace LibUSearch
{
    using usearch_index_t = System.IntPtr;
    using usearch_key_t = System.UInt64;
    using usearch_distance_t = System.Single;
    using usearch_error_t = System.String;

    public static class Interop
    {
        public delegate usearch_distance_t usearch_metric_t(object a, object b);

        public enum usearch_metric_kind_t
        {
            usearch_metric_ip_k = 0,
            usearch_metric_l2sq_k,
            usearch_metric_cos_k,
            usearch_metric_haversine_k,
            usearch_metric_pearson_k,
            usearch_metric_jaccard_k,
            usearch_metric_hamming_k,
            usearch_metric_tanimoto_k,
            usearch_metric_sorensen_k,
            usearch_metric_unknown_k,
        }

        public enum usearch_scalar_kind_t
        {
            usearch_scalar_f32_k = 0,
            usearch_scalar_f64_k,
            usearch_scalar_f16_k,
            usearch_scalar_f8_k,
            usearch_scalar_b1_k,
            usearch_scalar_unknown_k,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct usearch_init_options_t
        {
            public usearch_metric_kind_t metric_kind;
            public usearch_metric_t? metric;

            public usearch_scalar_kind_t quantization;
            public nuint dimensions;
            public nuint connectivity;
            public nuint expansion_add;
            public nuint expansion_search;
        }

        [DllImport("libusearch", EntryPoint = "usearch_init")]
        private static extern usearch_index_t _usearch_init(ref usearch_init_options_t options, out nint error);

        public static usearch_index_t usearch_init(ref usearch_init_options_t options, out usearch_error_t? error)
        {
            var result = _usearch_init(ref options, out var err);
            error = Marshal.PtrToStringAnsi(err);
            return result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_free")]
        private static extern void _usearch_free(usearch_index_t index, out nint error);

        public static void usearch_free(usearch_index_t index, out usearch_error_t? error)
        {
            _usearch_free(index, out var err);
            error = Marshal.PtrToStringAnsi(err);
        }

        [DllImport("libusearch", EntryPoint = "usearch_save")]
        private static extern void _usearch_save(usearch_index_t index, string path, out nint error);

        public static void usearch_save(usearch_index_t index, string path, out usearch_error_t? error)
        {
            _usearch_save(index, path, out var err);
            error = Marshal.PtrToStringAnsi(err);
        }

        [DllImport("libusearch", EntryPoint = "usearch_load")]
        private static extern void _usearch_load(usearch_index_t index, string path, out nint error);

        public static void usearch_load(usearch_index_t index, string path, out usearch_error_t? error)
        {
            _usearch_load(index, path, out var err);
            error = Marshal.PtrToStringAnsi(err);
        }

        [DllImport("libusearch", EntryPoint = "usearch_view")]
        private static extern void _usearch_view(usearch_index_t index, string path, out nint error);

        public static void usearch_view(usearch_index_t index, string path, out usearch_error_t? error)
        {
            _usearch_view(index, path, out var err);
            error = Marshal.PtrToStringAnsi(err);
        }

        [DllImport("libusearch", EntryPoint = "usearch_size")]
        private static extern nuint _usearch_size(usearch_index_t index, out nint error);

        public static int usearch_size(usearch_index_t index, out usearch_error_t? error)
        {
            var result = _usearch_size(index, out var err);
            error = Marshal.PtrToStringAnsi(err);
            return (int)result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_capacity")]
        private static extern nuint _usearch_capacity(usearch_index_t index, out nint error);

        public static int usearch_capacity(usearch_index_t index, out usearch_error_t? error)
        {
            var result = _usearch_capacity(index, out var err);
            error = Marshal.PtrToStringAnsi(err);
            return (int)result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_dimensions")]
        private static extern nuint _usearch_dimensions(usearch_index_t index, out nint error);

        public static int usearch_dimensions(usearch_index_t index, out usearch_error_t? error)
        {
            var result = _usearch_dimensions(index, out var err);
            error = Marshal.PtrToStringAnsi(err);
            return (int)result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_connectivity")]
        private static extern nuint _usearch_connectivity(usearch_index_t index, out nint error);

        public static nuint usearch_connectivity(usearch_index_t index, out usearch_error_t? error)
        {
            var result = _usearch_connectivity(index, out var err);
            error = Marshal.PtrToStringAnsi(err);
            return result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_reserve")]
        private static extern void _usearch_reserve(usearch_index_t index, nuint capacity, out nint error);

        public static void usearch_reserve(usearch_index_t index, int capacity, out usearch_error_t? error)
        {
            _usearch_reserve(index, (nuint)capacity, out var err);
            error = Marshal.PtrToStringAnsi(err);
        }

        [DllImport("libusearch", EntryPoint = "usearch_add")]
        private static extern void _usearch_add_f32(usearch_index_t index, usearch_key_t key, float[] vector, usearch_scalar_kind_t vector_kind, out nint error);

        [DllImport("libusearch", EntryPoint = "usearch_add")]
        private static extern void _usearch_add_f64(usearch_index_t index, usearch_key_t key, double[] vector, usearch_scalar_kind_t vector_kind, out nint error);

        public static void usearch_add<T>(usearch_index_t index, usearch_key_t key, T[] vector, out usearch_error_t? error) where T : struct
        {
            nint err;

            if (typeof(T) == typeof(float))
            {
                _usearch_add_f32(
                    index,
                    key,
                    (float[])(object)vector,
                    usearch_scalar_kind_t.usearch_scalar_f32_k,
                    out err);
            }
            else if (typeof(T) == typeof(double))
            {
                _usearch_add_f64(
                    index,
                    key,
                    (double[])(object)vector,
                    usearch_scalar_kind_t.usearch_scalar_f64_k,
                    out err);
            }
            else
            {
                throw new NotSupportedException($"Type not supported ({nameof(T)}).");
            }

            error = Marshal.PtrToStringAnsi(err);
        }

        [DllImport("libusearch", EntryPoint = "usearch_contains")]
        private static extern bool _usearch_contains(usearch_index_t index, usearch_key_t key, out nint error);

        public static bool usearch_contains(usearch_index_t index, usearch_key_t key, out usearch_error_t? error)
        {
            var result = _usearch_contains(index, key, out var err);
            error = Marshal.PtrToStringAnsi(err);
            return result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_search")]
        private static extern nuint _usearch_search_f32(
            usearch_index_t index,
            float[] query_vector,
            usearch_scalar_kind_t query_kind,
            nuint results_limit,
            usearch_key_t[] found_keys,
            usearch_distance_t[] found_distances,
            out nint error
        );

        [DllImport("libusearch", EntryPoint = "usearch_search")]
        private static extern nuint _usearch_search_f64(
            usearch_index_t index,
            double[] query_vector,
            usearch_scalar_kind_t query_kind,
            nuint results_limit,
            usearch_key_t[] found_keys,
            usearch_distance_t[] found_distances,
            out nint error
        );

        public static int usearch_search<T>(
            usearch_index_t index,
            T[] query_vector,
            int results_limit,
            out Dictionary<usearch_key_t, usearch_distance_t> found_labels_distances,
            out usearch_error_t? error
        ) where T : struct
        {
            int result;
            nint err;

            var found_labels = new usearch_key_t[results_limit];
            var found_distances = new usearch_distance_t[results_limit];

            if (typeof(T) == typeof(float))
            {
                result = (int)_usearch_search_f32(
                    index,
                    (float[])(object)query_vector,
                    usearch_scalar_kind_t.usearch_scalar_f32_k,
                    (nuint)results_limit,
                    found_labels,
                    found_distances,
                    out err);
            }
            else if (typeof(T) == typeof(double))
            {
                result = (int)_usearch_search_f64(
                    index,
                    (double[])(object)query_vector,
                    usearch_scalar_kind_t.usearch_scalar_f64_k,
                    (nuint)results_limit,
                    found_labels,
                    found_distances,
                    out err);
            }
            else
            {
                throw new NotSupportedException($"Type not supported ({nameof(T)}).");
            }

            found_labels_distances = new Dictionary<usearch_key_t, usearch_distance_t>();
            for (var i = 0; i < result; i++)
                found_labels_distances.Add(found_labels[i], found_distances[i]);

            error = Marshal.PtrToStringAnsi(err);
            return result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_get")]
        private static extern bool _usearch_get_f32(usearch_index_t index, usearch_key_t key, float[] vector, usearch_scalar_kind_t vector_kind, out nint error);

        [DllImport("libusearch", EntryPoint = "usearch_get")]
        private static extern bool _usearch_get_f64(usearch_index_t index, usearch_key_t key, double[] vector, usearch_scalar_kind_t vector_kind, out nint error);

        public static bool usearch_get<T>(usearch_index_t index, usearch_key_t key, T[] vector, out usearch_error_t? error) where T : struct
        {
            bool result;
            nint err;

            if (typeof(T) == typeof(float))
            {
                result = _usearch_get_f32(
                    index,
                    key,
                    (float[])(object)vector,
                    usearch_scalar_kind_t.usearch_scalar_f32_k,
                    out err);
            }
            else if (typeof(T) == typeof(double))
            {
                result = _usearch_get_f64(
                    index,
                    key,
                    (double[])(object)vector,
                    usearch_scalar_kind_t.usearch_scalar_f64_k,
                    out err);
            }
            else
            {
                throw new NotSupportedException($"Type not supported ({nameof(T)}).");
            }

            error = Marshal.PtrToStringAnsi(err);
            return result;
        }

        [DllImport("libusearch", EntryPoint = "usearch_remove")]
        private static extern void _usearch_remove(usearch_index_t index, usearch_key_t key, out nint error);

        public static void usearch_remove(usearch_index_t index, usearch_key_t key, out usearch_error_t? error)
        {
            _usearch_remove(index, key, out var err);
            error = Marshal.PtrToStringAnsi(err);
        }
    }
}
