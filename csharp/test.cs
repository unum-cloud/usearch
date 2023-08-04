using System.Diagnostics;
using System.Runtime.InteropServices;

namespace LibUSearch
{
    using static LibUSearch.Interop;
    using usearch_index_t = System.IntPtr;
    using usearch_label_t = System.UInt32;
    using usearch_distance_t = System.Single;
    using usearch_error_t = System.String;

    using f16_t = System.Half;
    using f32_t = System.Single;
    using f64_t = System.Double;

    static class Program
    {
        static void Main()
        {
            var tests = new Dictionary<string, Action>
            {
                ["f16_t"] = () => run_test<f16_t>("sample-f16.fvecs"),
                ["f32_t"] = () => run_test<f16_t>("sample-f32.fvecs"),
                ["f64_t"] = () => run_test<f16_t>("sample-f64.fvecs"),
            };

            foreach (var test in tests)
            {
                Console.WriteLine($"TESTS: {test.Key}");
                test.Value();
            }
        }

        static void run_test<T>(string file_path) where T : struct
        {
            parse_vectors_from_file<T>(file_path, out var vectors_count, out var vector_dimension, out var data);
            vectors_count = vectors_count > 100000 ? 100000 : vectors_count;

            // Tests
            test_init(vectors_count, vector_dimension);
            test_add_vector(data);
            test_get_vector(data);
            test_find_vector(data);
            test_remove_vector(data);
            test_save_load($"{file_path}.usearch_index.bin", data);
            test_view($"{file_path}.usearch_index.bin", data);
        }

        static void parse_vectors_from_file<T>(string file_path, out ulong count, out ulong dimension, out T[][] data) where T : struct
        {
            // Open the file (file format is one int32_t for dimension then array of vectors)
            var fi = new FileInfo(file_path);
            Debug.Assert(fi.Exists, "Failed to get metadata of the file");

            using FileStream fs = new FileStream(file_path, FileMode.Open, FileAccess.Read);
            using BinaryReader reader = new BinaryReader(fs);

            // Read vectors dimension and calculate count
            var ts = Marshal.SizeOf(typeof(T));
            var dim = reader.ReadInt32();
            Debug.Assert((fi.Length - sizeof(int)) % (dim * ts) == 0, "File does not contain a whole number of vectors");
            dimension = (ulong)dim;
            count = (ulong)((int)fi.Length / (dim * ts));

            // Allocate memory for the vectors' data
            data = new T[count][];
            Debug.Assert(data != null, "Failed to allocate memory");

            // Read the data
            for (var i = 0; i < (int)count; i++)
            {
                data[i] = new T[dim];

                var bytes = reader.ReadBytes(dim * ts);
                for (var j = 0; j < dim; j++)
                {
                    if (typeof(T) == typeof(f16_t))
                        data[i][j] = (T)(object)BitConverter.ToHalf(bytes, j * ts);
                    else if (typeof(T) == typeof(f32_t))
                        data[i][j] = (T)(object)BitConverter.ToSingle(bytes, j * ts);
                    else if (typeof(T) == typeof(f64_t))
                        data[i][j] = (T)(object)BitConverter.ToDouble(bytes, j * ts);
                }

                if (i == (int)count - 1)
                    break;
            }
        }

        static usearch_init_options_t create_options(ulong vector_dimension)
        {
            usearch_init_options_t opts;
            opts.connectivity = 2;
            opts.dimensions = (nuint)vector_dimension;
            opts.expansion_add = 40;
            opts.expansion_search = 16;
            opts.metric_kind = usearch_metric_kind_t.usearch_metric_cos_k;
            opts.metric = null;
            opts.quantization = usearch_scalar_kind_t.usearch_scalar_f32_k;
            return opts;
        }

        static void test_init(ulong vectors_count, ulong vector_dimension)
        {
            Console.Write("Test: Index Initialization...\n");

            usearch_init_options_t opts = create_options(vector_dimension);
            usearch_index_t idx = usearch_init(ref opts, out var error);
            Debug.Assert(error == null, error);
            usearch_free(idx, out error);
            Debug.Assert(error == null, error);

            idx = usearch_init(ref opts, out error);
            Debug.Assert(error == null, error);

            Debug.Assert(usearch_size(idx, out error) == 0, error);
            Debug.Assert(usearch_capacity(idx, out error) == 0, error);
            Debug.Assert(usearch_dimensions(idx, out error) == vector_dimension, error);
            Debug.Assert(usearch_connectivity(idx, out error) == opts.connectivity, error);

            usearch_reserve(idx, vectors_count, out error);
            Debug.Assert(error == null, error);
            Debug.Assert(usearch_size(idx, out error) == 0, error);
            Debug.Assert(usearch_capacity(idx, out error) == vectors_count, error);
            Debug.Assert(usearch_dimensions(idx, out error) == vector_dimension, error);
            Debug.Assert(usearch_connectivity(idx, out error) == opts.connectivity, error);

            usearch_free(idx, out error);
            Debug.Assert(error == null, error);

            Console.Write("Test: Index Initialization - PASSED\n");
        }

        static void test_add_vector<T>(T[][] data) where T : struct
        {
            Console.Write("Test: Add Vector...\n");

            var vectors_count = (ulong)data.Length;
            var vector_dimension = (ulong)data[0].Length;

            usearch_init_options_t opts = create_options(vector_dimension);
            usearch_index_t idx = usearch_init(ref opts, out var error);
            usearch_reserve(idx, vectors_count, out error);

            // Add vectors
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                usearch_add(idx, label, data[i], out error);
                Debug.Assert(error == null, error);
            }

            Debug.Assert(usearch_size(idx, out error) == vectors_count, error);
            Debug.Assert(usearch_capacity(idx, out error) == vectors_count, error);

            // Check vectors in the index
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                Debug.Assert(usearch_contains(idx, label, out error), error);
            }
            Debug.Assert(!usearch_contains(idx, unchecked((uint)-1), out error), error); // Non existing label

            usearch_free(idx, out error);
            Console.Write("Test: Add Vector - PASSED\n");
        }

        static void test_get_vector<T>(T[][] data) where T : struct
        {
            Console.Write("Test: Get Vector...\n");

            var vectors_count = (ulong)data.Length;
            var vector_dimension = (ulong)data[0].Length;

            usearch_init_options_t opts = create_options(vector_dimension);
            usearch_index_t idx = usearch_init(ref opts, out var error);
            usearch_reserve(idx, vectors_count, out error);

            // Add vectors
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                usearch_add(idx, label, data[i], out error);
                Debug.Assert(error == null, error);
            }

            Debug.Assert(usearch_size(idx, out error) == vectors_count, error);
            Debug.Assert(usearch_capacity(idx, out error) == vectors_count, error);

            // Get vectors from the index
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                Debug.Assert(usearch_get(idx, label, out T[]? vector, out error), error);
                Debug.Assert(vector?.SequenceEqual(data[i]) == true, "Vector does not match");
            }

            usearch_free(idx, out error);
            Console.Write("Test: Get Vector - PASSED\n");
        }

        static void test_find_vector<T>(T[][] data) where T : struct
        {
            Console.Write("Test: Find Vector...\n");

            var vectors_count = (ulong)data.Length;
            var vector_dimension = (ulong)data[0].Length;

            var opts = create_options(vector_dimension);
            var idx = usearch_init(ref opts, out var error);
            usearch_reserve(idx, vectors_count, out error);

            // Add vectors
            for (ulong i = 0; i < vectors_count; i++)
            {
                var label = (usearch_label_t)i;
                usearch_add(idx, label, data[i], out error);
                Debug.Assert(error == null, error);
            }

            // Find the vectors
            for (uint i = 0; i < vectors_count; i++)
            {
                var query_vector = data[i];
                var found_count = usearch_search(idx, query_vector, vectors_count, out var labels_distances, out error);
                Debug.Assert(error == null, error);
                Debug.Assert(found_count == vectors_count, "Vector is missing");
            }

            usearch_free(idx, out error);
            Console.Write("Test: Find Vector - PASSED\n");
        }

        static void test_remove_vector<T>(T[][] data) where T : struct
        {
            Console.Write("Test: Remove Vector...\n");

            var vectors_count = (ulong)data.Length;
            var vector_dimension = (ulong)data[0].Length;

            usearch_init_options_t opts = create_options(vector_dimension);
            usearch_index_t idx = usearch_init(ref opts, out var error);
            usearch_reserve(idx, vectors_count, out error);

            // Add vectors
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                usearch_add(idx, label, data[i], out error);
                Debug.Assert(error == null, error);
            }

            // Remove the vectors
            for (uint i = 0; i < vectors_count; i++)
            {
                usearch_label_t label = i;
                usearch_remove(idx, label, out error);
                Debug.Assert(error == null, error);
            }

            usearch_free(idx, out error);
            Console.Write("Test: Remove Vector - PASSED\n");
        }

        static void test_save_load<T>(string file_path, T[][] data) where T : struct
        {
            Console.Write("Test: Save/Load...\n");

            var vectors_count = (ulong)data.Length;
            var vector_dimension = (ulong)data[0].Length;

            usearch_init_options_t opts = create_options(vector_dimension);
            usearch_index_t idx = usearch_init(ref opts, out var error);
            usearch_reserve(idx, vectors_count, out error);

            // Add vectors
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                usearch_add(idx, label, data[i], out error);
                Debug.Assert(error == null, error);
            }

            // Save and free the index
            usearch_save(idx, file_path, out error);
            Debug.Assert(error == null, error);
            usearch_free(idx, out error);
            Debug.Assert(error == null, error);

            // Reinit
            idx = usearch_init(ref opts, out error);
            Debug.Assert(error == null, error);
            Debug.Assert(usearch_size(idx, out error) == 0, error);

            // Load
            usearch_load(idx, file_path, out error);
            Debug.Assert(error == null, error);
            Debug.Assert(usearch_size(idx, out error) == vectors_count, error);
            Debug.Assert(usearch_capacity(idx, out error) == vectors_count, error);
            Debug.Assert(usearch_dimensions(idx, out error) == vector_dimension, error);
            Debug.Assert(usearch_connectivity(idx, out error) == opts.connectivity, error);

            // Check vectors in the index
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                Debug.Assert(usearch_contains(idx, label, out error), error);
            }

            usearch_free(idx, out error);
            Console.Write("Test: Save/Load - PASSED\n");
        }

        static void test_view<T>(string file_path, T[][] data) where T : struct
        {
            Console.Write("Test: View...\n");

            var vectors_count = (ulong)data.Length;
            var vector_dimension = (ulong)data[0].Length;

            var opts = create_options(vector_dimension);
            var idx = usearch_init(ref opts, out var error);
            usearch_reserve(idx, vectors_count, out error);

            // Add vectors
            for (uint i = 0; i < vectors_count; ++i)
            {
                usearch_label_t label = i;
                usearch_add(idx, label, data[i], out error);
                Debug.Assert(error == null, error);
            }

            // Save and free the index
            usearch_save(idx, file_path, out error);
            Debug.Assert(error == null, error);
            usearch_free(idx, out error);
            Debug.Assert(error == null, error);

            // Reinit
            idx = usearch_init(ref opts, out error);
            Debug.Assert(error == null, error);

            // View
            usearch_view(idx, file_path, out error);
            Debug.Assert(error == null, error);

            usearch_free(idx, out error);
            Console.Write("Test: View - PASSED\n");
        }
    }
}
