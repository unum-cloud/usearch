using Cloud.Unum.USearch;

namespace Cloud.Unum.USearch.Tests;

public class InitTests
{
    // Epsilon for real values comparisons
    private const float Epsilon = 0.000001f;

    [Fact]
    public void InitializedAndDisposedSucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3))
        {
            Assert.NotNull(index);
            Assert.Equal((uint)3, index.Dimensions());
        }
    }

    [Fact]
    public void AddDoubleVectorSucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float64, 3))
        {
            index.Add(1, new double[] { 1.0, 2.0, 3.0 });
            Assert.True(index.Contains(1));
            Assert.False(index.Contains(2));
            Assert.Equal((uint)1, index.Size());
        }
    }

    [Fact]
    public void AddDoubleVectorsUnderSameKeySucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float64, 3, multi: true))
        {
            index.Add(1, new double[] { 1.0, 2.0, 3.0 });
            index.Add(2, new double[] { 1.0, 2.0, 3.0 });
            index.Add(3, new double[] { 1.0, 2.0, 3.0 });
            index.Add(1, new double[] { 1.0, 2.0, 3.0 });
            index.Add(2, new double[] { 1.0, 2.0, 3.0 });
            Assert.True(index.Contains(1));
            Assert.True(index.Contains(2));
            Assert.True(index.Contains(3));
            Assert.Equal((uint)5, index.Size());
        }
    }

    [Fact]
    public void ItCanGetMultipleFloatVectorsForSingleKey()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3, multi: true))
        {
            var inputVectors = new float[][] {
                    new float[] { 1.0f, 6.0f, 11.0f },
                    new float[] { 2.0f, 7.0f, 12.0f },
                    new float[] { 3.0f, 8.0f, 13.0f },
                    new float[] { 5.0f, 10.0f, 15.0f }
            };
            index.Add(
                new ulong[] { 2, 2, 2, 2 },
                inputVectors
            );
            Assert.True(index.Contains(2));
            Assert.Equal((uint)4, index.Size());
            Assert.Equal(4, index.Count(2));

            int foundVectorsCount = index.Get(2, 3, out float[][] retrievedVectors);
            Assert.Equal(3, foundVectorsCount);
        }
    }

    [Fact]
    public void ItCanGetMultipleDoubleVectorsForSingleKey()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float64, 3, multi: true))
        {
            var inputVectors = new double[][] {
                new double[] { 1.0, 6.0, 11.0 },
                new double[] { 2.0, 7.0, 12.0 },
                new double[] { 3.0, 8.0, 13.0 },
                new double[] { 5.0, 10.0, 15.0 }
            };
            index.Add(
                new ulong[] { 2, 2, 2, 2 },
                inputVectors
            );
            Assert.True(index.Contains(2));
            Assert.Equal((uint)4, index.Size());
            Assert.Equal(4, index.Count(2));

            int foundVectorsCount = index.Get(2, 3, out double[][] retrievedVectors);
            Assert.Equal(3, foundVectorsCount);
        }
    }

    [Fact]
    public void AddFloatVectorSucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3))
        {
            index.Add(1, new float[] { 1.0f, 2.0f, 3.0f });
            Assert.Equal((uint)1, index.Size());
            Assert.True(index.Contains(1));
            Assert.False(index.Contains(2));
        }
    }

    [Fact]
    public void AddFloatVectorsUnderSameKeySucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3, multi: true))
        {
            index.Add(1, new float[] { 1.0f, 2.0f, 3.0f });
            index.Add(2, new float[] { 1.0f, 2.0f, 3.0f });
            index.Add(3, new float[] { 1.0f, 2.0f, 3.0f });
            index.Add(1, new float[] { 1.0f, 2.0f, 3.0f });
            index.Add(2, new float[] { 1.0f, 2.0f, 3.0f });
            Assert.True(index.Contains(1));
            Assert.True(index.Contains(2));
            Assert.True(index.Contains(3));
            Assert.Equal((uint)5, index.Size());
        }
    }

    [Fact]
    public void AddGetFloatVectorSucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3))
        {
            var inputVector = new float[] { 1.0f, 2.0f, 3.0f };
            index.Add(1, inputVector);
            Assert.Equal((uint)1, index.Size());
            Assert.True(index.Contains(1));
            Assert.False(index.Contains(2));
            Assert.True(index.Get(1, out float[] retrievedVector) == 1);
            Assert.NotNull(retrievedVector);
            bool areEqual = inputVector
                .Zip(retrievedVector, (a, b) => (a, b))
                .All(pair => Math.Abs(pair.a - pair.b) <= Epsilon);

            Assert.True(areEqual);
        }
    }

    [Fact]
    public void AddGetDoubleVectorSucceeds()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3))
        {
            var inputVector = new double[] { 1.0, 2.0, 3.33 };
            index.Add(1, inputVector);
            Assert.Equal((uint)1, index.Size());
            Assert.True(index.Contains(1));
            Assert.False(index.Contains(2));
            Assert.True(index.Get(1, out double[] retrievedVector) == 1);
            Assert.NotNull(retrievedVector);
            bool areEqual = inputVector
                .Zip(retrievedVector, (a, b) => (a, b))
                .All(pair => Math.Abs(pair.a - pair.b) <= Epsilon);

            Assert.True(areEqual);
        }
    }

    [Fact]
    public void ItThrowsException_AddAddWithSameKeyAndGetVector()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 3))
        {
            ulong key = 1;
            float[] vector1 = { 1.0f, 2.0f, 33.3f };
            index.Add(key, vector1);
            float[] vector2 = { 3.0f, 4.0f, 34.2f };
            Assert.Throws<USearchException>(() => index.Add(key, vector2));
        }
    }

    [Fact]
    public void TestAddRemoveAddAndGet()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 2))
        {
            ulong key = 1;
            float[] vector = { 1.0f, 2.0f };

            index.Add(key, vector);
            Assert.True(index.Contains(key));
            index.Remove(key);
            Assert.False(index.Contains(key));
            index.Add(key, vector);
            Assert.True(index.Contains(key));
            Assert.True(index.Get(key, out float[] retrievedVector) == 1);
            Assert.Equal(vector, retrievedVector);
        }
    }

    [Fact]
    public void TestAddRemoveAddWithSameKeyVectorAndGet()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 2))
        {
            ulong key = 1;
            float[] vector = { 1.0f, 2.0f };

            index.Add(key, vector);
            Assert.True(index.Contains(key));
            index.Remove(key);
            Assert.False(index.Contains(key));
            index.Add(key, vector);
            Assert.True(index.Contains(key));
            Assert.True(index.Get(key, out float[] retrievedVector) == 1);
            Assert.Equal(vector, retrievedVector);
        }
    }

    [Fact]
    public void TestGetKeyThatNotExist()
    {
        using (var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, 2))
        {
            Assert.True(index.Get(1, out float[] retrievedVector) == 0);
            Assert.False(index.Contains(1));
        }
    }

    [Fact]
    public void Search_FindsExpectedResults()
    {
        using (var index = new USearchIndex(MetricKind.L2sq, ScalarKind.Float32, 2))
        {
            index.Add(
                Enumerable.Range(1, 5).Select(i => (ulong)i).ToArray(),
                new float[][]
                {
                    new float[] { -0.5f, -0.6f },
                    new float[] { -0.1f, -0.1f },
                    new float[] { 0.2f, 0.2f },
                    new float[] { 0.7f, 0.8f },
                    new float[] { 0.9f, 1.0f }
                }
            );
            var queryVector = new float[] { 0.0f, 0.0f };

            var matches = index.Search(queryVector, 2, out ulong[] keys, out float[] distances);

            Assert.Equal(2, matches);
            Assert.Contains(2UL, keys);
            Assert.Contains(3UL, keys);
            Assert.Equal(0.02, distances[0], Epsilon);
            Assert.Equal(0.08, distances[1], Epsilon);
        }
    }

    [Fact]
    public void Search_AfterRemovingKey_FindsExpectedResults()
    {
        using (var index = new USearchIndex(MetricKind.L2sq, ScalarKind.Float32, 2))
        {
            index.Add(1, new float[] { 0.1f, 0.2f });
            index.Add(2, new float[] { 0.2f, 0.3f });
            index.Add(3, new float[] { 0.3f, 0.4f });
            index.Remove(2);
            var queryVector = new float[] { 0.2f, 0.3f };

            var matches = index.Search(queryVector, 2, out ulong[] keys, out float[] distances);

            Assert.Equal(2, matches);
            Assert.Contains(1UL, keys);
            Assert.Contains(3UL, keys);
            Assert.DoesNotContain(2UL, keys);
        }
    }

    [Fact]
    public void Search_WithVectorsLessThanResultsLimit()
    {
        using (var index = new USearchIndex(MetricKind.L2sq, ScalarKind.Float32, 2))
        {
            index.Add(1, new float[] { 0.9f, 0.9f });
            index.Add(2, new float[] { 0.8f, 0.8f });
            var queryVector = new float[] { 0.1f, 0.2f };

            var matches = index.Search(queryVector, 10, out ulong[] keys, out float[] distances);

            Assert.Equal(2, matches);
            Assert.Contains(1UL, keys);
            Assert.Contains(2UL, keys);
        }
    }

    [Fact]
    public void ItCanRenameKey()
    {
        using (var index = new USearchIndex(MetricKind.L2sq, ScalarKind.Float32, 2))
        {
            index.Add(1, new float[] { 0.9f, 0.9f });
            index.Add(2, new float[] { 0.8f, 0.8f });
            index.Rename(1, 3);
            Assert.False(index.Contains(1));
            Assert.True(index.Contains(2));
            Assert.True(index.Contains(3));
        }
    }

}
