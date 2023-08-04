using System.Diagnostics;

using static LibUSearch.Interop;
using usearch_index_t = System.IntPtr;
using usearch_label_t = System.UInt32;
using usearch_distance_t = System.Single;
using usearch_error_t = System.String;

class Program
{
    static unsafe void Main()
    {
        parse_vectors_from_file(@"sample.fvecs", out nuint vectors_count, out nuint vector_dimension, out var data);
        vectors_count = vectors_count > 100000 ? 100000 : vectors_count;

        fixed (float* ptr = data)
        {
            // Test
            test_init(vectors_count, vector_dimension);
            test_add_vector(vectors_count, vector_dimension, ptr);
            test_find_vector(vectors_count, vector_dimension, ptr);
            test_remove_vector(vectors_count, vector_dimension, ptr);
            test_save_load(vectors_count, vector_dimension, ptr);
            test_view(vectors_count, vector_dimension, ptr);
        }
    }

    static void parse_vectors_from_file(string file_path, out nuint count, out nuint dimension, out float[] data)
    {
        // Open the file
        var fi = new FileInfo(file_path);
        Debug.Assert(fi.Exists, "Failed to get metadata of the file");

        using FileStream fs = new FileStream(file_path, FileMode.Open, FileAccess.Read);
        using BinaryReader reader = new BinaryReader(fs);

        // Read vectors dimension and calculate count
        var dim = reader.ReadInt32();
        Debug.Assert(fi.Length % ((dim + 1) * sizeof(float)) == 0, "File does not contain a whole number of vectors");
        dimension = (nuint)dim;
        count = (nuint)(fi.Length / ((dim + 1) * sizeof(float)));

        // Allocate memory for the vectors' data
        data = new float[count * dimension];
        Debug.Assert(data != null, "Failed to allocate memory");

        // Read the data
        for (nuint i = 0; i < count; ++i)
        {
            var bytes = reader.ReadBytes((int)dimension * sizeof(float));
            Buffer.BlockCopy(bytes, 0, data ?? new float[0], (int)(i * dimension * sizeof(float)), bytes.Length);

            if (i == count - 1)
                break;

            // Skip
            reader.ReadSingle();
        }
    }

    static usearch_init_options_t create_options(nuint vector_dimension)
    {
        usearch_init_options_t opts;
        opts.connectivity = 2;
        opts.dimensions = vector_dimension;
        opts.expansion_add = 40;
        opts.expansion_search = 16;
        opts.metric_kind = usearch_metric_kind_t.usearch_metric_cos_k;
        opts.metric = null;
        opts.quantization = usearch_scalar_kind_t.usearch_scalar_f32_k;
        return opts;
    }

    static void test_init(nuint vectors_count, nuint vector_dimension)
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

    static unsafe void test_add_vector(nuint vectors_count, nuint vector_dimension, float* data)
    {
        Console.Write("Test: Add Vector...\n");

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data + i * vector_dimension, usearch_scalar_kind_t.usearch_scalar_f32_k, out error);
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

    static unsafe void test_find_vector(nuint vectors_count, nuint vector_dimension, float* data)
    {
        Console.Write("Test: Find Vector...\n");

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Create result buffers
        nuint results_count = Math.Min(vectors_count, 10);
        usearch_label_t[] labels = new usearch_label_t[results_count];
        float[] distances = new float[results_count];
        Debug.Assert(labels != null && distances != null, "Failed to allocate memory");

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data + i * vector_dimension, usearch_scalar_kind_t.usearch_scalar_f32_k, out error);
            Debug.Assert(error == null, error);
        }

        // Find the vectors
        for (uint i = 0; i < vectors_count; i++)
        {
            void* query_vector = data + i * vector_dimension;
            nuint found_count =
                usearch_search(idx, query_vector, usearch_scalar_kind_t.usearch_scalar_f32_k, results_count, labels ?? new usearch_label_t[0], distances ?? new usearch_distance_t[0], out error);
            Debug.Assert(error == null, error);
            Debug.Assert((found_count = results_count) > 0, "Vector is missing");
        }

        usearch_free(idx, out error);
        Console.Write("Test: Find Vector - PASSED\n");
    }

    static unsafe void test_remove_vector(nuint vectors_count, nuint vector_dimension, float* data)
    {
        Console.Write("Test: Remove Vector...\n");

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data + i * vector_dimension, usearch_scalar_kind_t.usearch_scalar_f32_k, out error);
            Debug.Assert(error == null, error);
        }

        // Remove the vectors
        for (uint i = 0; i < vectors_count; i++)
        {
            usearch_label_t label = i;
            usearch_remove(idx, label, out error);
            Debug.Assert(!usearch_error_t.IsNullOrEmpty(error), "Currently, Remove is not supported");
        }

        usearch_free(idx, out error);
        Console.Write("Test: Remove Vector - PASSED\n");
    }

    static unsafe void test_save_load(nuint vectors_count, nuint vector_dimension, float* data)
    {
        Console.Write("Test: Save/Load...\n");

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data + i * vector_dimension, usearch_scalar_kind_t.usearch_scalar_f32_k, out error);
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

    static unsafe void test_view(nuint vectors_count, nuint vector_dimension, float* data)
    {
        Console.Write("Test: View...\n");

        usearch_init_options_t opts = create_options(vector_dimension);
        usearch_index_t idx = usearch_init(ref opts, out usearch_error_t? error);
        usearch_reserve(idx, vectors_count, out error);

        // Add vectors
        for (uint i = 0; i < vectors_count; ++i)
        {
            usearch_label_t label = i;
            usearch_add(idx, label, data + i * vector_dimension, usearch_scalar_kind_t.usearch_scalar_f32_k, out error);
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
