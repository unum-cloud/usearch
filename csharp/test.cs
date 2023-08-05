using System.Diagnostics;

using static LibUSearch.Interop;
using usearch_index_t = System.IntPtr;
using usearch_label_t = System.UInt32;
using usearch_distance_t = System.Single;
using usearch_error_t = System.String;

class Program
{
    static void Main()
    {
        parse_vectors_from_file(@"sample.fvecs", out var vectors_count, out var vector_dimension, out var data);
        vectors_count = vectors_count > 100000 ? 100000 : vectors_count;

        // Test
        test_init(vectors_count, vector_dimension);
        test_add_vector(data);
        test_find_vector(data);
        test_remove_vector(data);
        test_save_load(data);
        test_view(data);
    }

    static void parse_vectors_from_file(string file_path, out int count, out int dimension, out float[][] data)
    {
        // Open the file
        var fi = new FileInfo(file_path);
        Debug.Assert(fi.Exists, "Failed to get metadata of the file");

        using FileStream fs = new FileStream(file_path, FileMode.Open, FileAccess.Read);
        using BinaryReader reader = new BinaryReader(fs);

        // Read vectors dimension and calculate count
        var dim = reader.ReadInt32();
        Debug.Assert(fi.Length % ((dim + 1) * sizeof(float)) == 0, "File does not contain a whole number of vectors");
        dimension = dim;
        count = (int)fi.Length / ((dim + 1) * sizeof(float));

        // Allocate memory for the vectors' data
        data = new float[count][];
        Debug.Assert(data != null, "Failed to allocate memory");

        // Read the data
        for (var i = 0; i < count; ++i)
        {
            data[i] = new float[dimension];

            var bytes = reader.ReadBytes(dimension * sizeof(float));
            Buffer.BlockCopy(bytes, 0, data[i], 0, dimension * sizeof(float));

            if (i == count - 1)
                break;

            // Skip
            reader.ReadSingle();
        }
    }

    static usearch_init_options_t create_options(int vector_dimension)
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

    static void test_init(int vectors_count, int vector_dimension)
    {
        Console.Write("Test: Index Initialization...\n");

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
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

        var vectors_count = data.Length;
        var vector_dimension = data[0].Length;

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
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

    static void test_find_vector<T>(T[][] data) where T : struct
    {
        Console.Write("Test: Find Vector...\n");

        var vectors_count = data.Length;
        var vector_dimension = data[0].Length;

        var opts = create_options(vector_dimension);
        var idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (var i = 0; i < vectors_count; i++)
        {
            var label = (usearch_label_t)i + 1;
            usearch_add(idx, label, data[i], out error);
            Debug.Assert(error == null, error);
        }

        // Find the vectors
        for (uint i = 0; i < vectors_count; i++)
        {
            var query_vector = data[i];
            var found_count = usearch_search(idx, query_vector, vectors_count, out var labels_distances, out error);
            Debug.Assert(error == null, error);
            //Debug.Assert(found_count == vectors_count, "Vector is missing");
        }

        usearch_free(idx, out error);
        Console.Write("Test: Find Vector - PASSED\n");
    }

    static void test_remove_vector<T>(T[][] data) where T : struct
    {
        Console.Write("Test: Remove Vector...\n");

        var vectors_count = data.Length;
        var vector_dimension = data[0].Length;

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
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

    static void test_save_load<T>(T[][] data) where T : struct
    {
        Console.Write("Test: Save/Load...\n");

        var vectors_count = data.Length;
        var vector_dimension = data[0].Length;

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data[i], out error);
            Debug.Assert(error == null, error);
        }

        // Save and free the index
        usearch_save(idx, "usearch_index.bin", out error);
        Debug.Assert(error == null, error);
        usearch_free(idx, out error);
        Debug.Assert(error == null, error);

        // Reinit
        idx = usearch_init(ref opts, out error);
        Debug.Assert(error == null, error);
        Debug.Assert(usearch_size(idx, out error) == 0, error);

        // Load
        usearch_load(idx, "usearch_index.bin", out error);
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

    static void test_view<T>(T[][] data) where T : struct
    {
        Console.Write("Test: View...\n");

        var vectors_count = data.Length;
        var vector_dimension = data[0].Length;

        var opts = create_options(vector_dimension);
        var idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data[i], out error);
            Debug.Assert(error == null, error);
        }

        // Save and free the index
        usearch_save(idx, "usearch_index.bin", out error);
        Debug.Assert(error == null, error);
        usearch_free(idx, out error);
        Debug.Assert(error == null, error);

        // Reinit
        idx = usearch_init(ref opts, out error);
        Debug.Assert(error == null, error);

        // View
        usearch_view(idx, "usearch_index.bin", out error);
        Debug.Assert(error == null, error);

        usearch_free(idx, out error);
        Console.Write("Test: View - PASSED\n");
    }
}
