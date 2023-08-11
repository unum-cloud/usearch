namespace LibUsearch.Tests;

public class InitTests
{
    // Epsilon for real values comparings
    private const float Epsilon = 0.000001f;

    [Fact]
    public void InitializedAndDisposedSucceeds()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            3
        );
        Assert.NotNull(uSearch);
        Assert.Equal((uint)3, uSearch.Dimensions());
        uSearch.Dispose();
    }

    [Fact]
    public void AddDoubleVectorSucceeds()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f64_k,
            3
        );
        uSearch.Add(1, new double[] { 1.0, 2.0, 3.0 });
        Assert.True(uSearch.Contains(1));
        Assert.False(uSearch.Contains(2));
        Assert.Equal((uint)1, uSearch.Size());
        uSearch.Dispose();
    }

    [Fact]
    public void AddFloatVectorSucceeds()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            3
        );
        uSearch.Add(1, new float[] { 1.0f, 2.0f, 3.0f });
        Assert.Equal((uint)1, uSearch.Size());
        Assert.True(uSearch.Contains(1));
        Assert.False(uSearch.Contains(2));
        uSearch.Dispose();
    }

    [Fact]
    public void AddGetFloatVectorSucceeds()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            3
        );
        var inputVector = new float[] { 1.0f, 2.0f, 3.0f };
        uSearch.Add(1, inputVector);
        Assert.Equal((uint)1, uSearch.Size());
        Assert.True(uSearch.Contains(1));
        Assert.False(uSearch.Contains(2));
        Assert.True(uSearch.Get(1, out float[] retrievedVector));
        Assert.NotNull(retrievedVector);
        bool areEqual = inputVector
                                .Zip(retrievedVector, (a, b) => (a, b))
                                .All(pair => Math.Abs(pair.a - pair.b) <= Epsilon);

        Assert.True(areEqual);

        uSearch.Dispose();
    }

    [Fact]
    public void AddGetDoubleVectorSucceeds()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f64_k,
            3
        );
        var inputVector = new double[] { 1.0, 2.0, 3.33 };
        uSearch.Add(1, inputVector);
        Assert.Equal((uint)1, uSearch.Size());
        Assert.True(uSearch.Contains(1));
        Assert.False(uSearch.Contains(2));
        Assert.True(uSearch.Get(1, out double[] retrievedVector));
        Assert.NotNull(retrievedVector);
        bool areEqual = inputVector
                                .Zip(retrievedVector, (a, b) => (a, b))
                                .All(pair => Math.Abs(pair.a - pair.b) <= Epsilon);

        Assert.True(areEqual);

        uSearch.Dispose();
    }


    // TODO return back this test when std::runtime_error wont be thrown from USearch lib
    [Fact(Skip = "This test is temporarily disabled because of hanging.")]
    public void ItThrowsException_AddAddWithSameKeyAndGetVector()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            3
        );

        ulong key = 1;
        float[] vector1 = { 1.0f, 2.0f, 33.3f };
        uSearch.Add(key, vector1);
        float[] vector2 = { 3.0f, 4.0f, 34.2f };
        Assert.Throws<USearchIndex.USearchException>(() => uSearch.Add(key, vector2));

        uSearch.Dispose();
    }

    [Fact]
    public void TestAddRemoveAddAndGet()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            2
        );
        try
        {
            ulong key = 1;
            float[] vector = { 1.0f, 2.0f };

            uSearch.Add(key, vector);
            Assert.True(uSearch.Contains(key));
            uSearch.Remove(key);
            Assert.False(uSearch.Contains(key));
            uSearch.Add(key, vector);
            Assert.True(uSearch.Contains(key));
            Assert.True(uSearch.Get(key, out float[] retrievedVector));
            Assert.Equal(vector, retrievedVector);
        }
        finally
        {
            uSearch.Dispose();
        }
    }

    [Fact]
    public void TestAddRemoveAddWithSameKeyVectorAndGet()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            2
        );
        try
        {
            ulong key = 1;
            float[] vector = { 1.0f, 2.0f };

            uSearch.Add(key, vector);
            Assert.True(uSearch.Contains(key));
            uSearch.Remove(key);
            Assert.False(uSearch.Contains(key));
            uSearch.Add(key, vector);
            Assert.True(uSearch.Contains(key));
            Assert.True(uSearch.Get(key, out float[] retrievedVector));
            Assert.Equal(vector, retrievedVector);
        }
        finally
        {
            uSearch.Dispose();
        }
    }

    //[Fact(Skip = "This test is temporarily disabled because of hanging.")]
    [Fact]
    public void TestGetKeyThatNotExist()
    {
        var uSearch = new USearchIndex(
            usearch_metric_kind_t.usearch_metric_cos_k,
            usearch_scalar_kind_t.usearch_scalar_f32_k,
            3
        );
        try
        {
            Assert.False(uSearch.Get(1, out float[] retrievedVector));
            Assert.False(uSearch.Contains(1));
        }
        finally
        {
            uSearch.Dispose();
        }
    }


}
