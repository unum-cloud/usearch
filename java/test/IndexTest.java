import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import org.junit.AfterClass;
import org.junit.Test;

import cloud.unum.usearch.Index;

public class IndexTest {
    public static void deleteDirectoryFiles(String path) {
        File directory = new File(path);
        if (!directory.isDirectory())
            return;

        for (File f : directory.listFiles())
            f.delete();
    }

    @Test
    public void test() {
        String path = "./tmp/";
        deleteDirectoryFiles(path);

        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            float vec[] = { 10, 20 };
            index.reserve(10);
            index.add(42, vec);
            long[] keys = index.search(vec, 5);
        }

    }

    @AfterClass
    public static void tearDown() {
        System.out.println("Java Tests Passed!");
    }

    @Test
    public void testGetSuccess() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            float vec[] = { 10, 20 };
            index.reserve(10);
            index.add(42, vec);

            assertArrayEquals(vec, index.get(42), 0.01f);
        }
    }

    @Test
    public void testGetFailed() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            float vec[] = { 10, 20 };
            index.reserve(10);
            index.add(42, vec);

            assertThrows(IllegalArgumentException.class, () -> index.get(41));
        }
    }

    @Test
    public void testUseAfterClose() {
        Index index = new Index.Config().metric("cos").dimensions(2).build();
        float vec[] = { 10, 20 };
        index.reserve(10);
        index.add(42, vec);
        assertEquals(1, index.size());
        index.close();
        assertThrows(IllegalStateException.class, () -> index.size());
    }

    @Test
    public void testLoadFromPath() throws IOException {
        File indexFile = File.createTempFile("test", "uidx");

        float vec[] = { 10, 20 };
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            index.reserve(10);
            index.add(42, vec);
            index.save(indexFile.getAbsolutePath());
        }

        try (Index index = Index.loadFromPath(indexFile.getAbsolutePath())) {
            assertArrayEquals(vec, index.get(42), 0.01f);
        }
    }

    @Test
    public void testLargeVectors() throws IOException {
        File indexFile = File.createTempFile("test", "uidx");

        int dimensions = 256;
        int numVectors = 100;
        try (Index index = new Index.Config().metric("cos").dimensions(dimensions).build()) {
            index.reserve(numVectors);
            for (int v = 0; v < numVectors; v++) {
                index.add(v + 1, randomVector(dimensions));
            }
            index.save(indexFile.getAbsolutePath());
        }

        try (Index index = Index.loadFromPath(indexFile.getAbsolutePath())) {
            for (int i = 0; i < 100; i++) {
                long[] keys = index.search(randomVector(dimensions), 10);
                for (long key : keys) {
                    assertTrue(key >= 1 && key <= numVectors);
                }
            }
        }
    }

    @Test
    public void testShortResults() throws IOException {
        int dimensions = 256;
        int numVectors = 5;
        try (Index index = new Index.Config().metric("cos").dimensions(dimensions).build()) {
            index.reserve(numVectors);
            for (int v = 0; v < numVectors; v++) {
                index.add(v + 1, randomVector(dimensions));
            }

            long[] keys = index.search(randomVector(dimensions), numVectors + 100);
            assertEquals(numVectors, keys.length);
            for (long key : keys) {
                assertNotEquals(0, key);
            }
        }
    }

    private static float[] randomVector(int dimensions) {
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = ThreadLocalRandom.current().nextFloat(2.f);
        }
        return vector;
    }

    @Test
    public void testMemoryUsage() {
        try (Index index = new Index.Config().metric("cos").dimensions(256).build()) {
            // Test empty index
            long initialMemory = index.memoryUsage();
            assertTrue("Initial memory usage should be positive", initialMemory > 0);

            // Add some vectors
            index.reserve(1000);
            long afterReserve = index.memoryUsage();
            assertTrue("Memory should increase after reserve", afterReserve >= initialMemory);

            // Add vectors
            float[] vector = new float[256];
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 256; j++) {
                    vector[j] = (float) Math.random();
                }
                index.add(i, vector);
            }

            long afterAdding = index.memoryUsage();
            assertTrue("Memory should increase after adding vectors", afterAdding > afterReserve);

            // Memory should be reasonable (not too small, not too large)
            assertTrue("Memory usage should be reasonable",
                    afterAdding > 1000 && afterAdding < 1_000_000_000L);
        }
    }

    @Test
    public void testDoubleVectorWithFloat64() {
        try (Index index = new Index.Config().metric("cos").dimensions(3).quantization("f64").build()) {
            double[] vec = { 1.0, 2.0, 3.0 };
            index.reserve(10);
            index.add(42, vec);

            double[] retrieved = new double[3];
            index.getInto(42, retrieved);
            assertArrayEquals(vec, retrieved, 0.01);
        }
    }

    @Test
    public void testByteVectorWithInt8() {
        try (Index index = new Index.Config().metric("cos").dimensions(4).quantization("i8").build()) {
            byte[] vec = { 10, 20, 30, 40 };
            index.reserve(10);
            index.add(42, vec);

            byte[] retrieved = new byte[4];
            index.getInto(42, retrieved);
            assertArrayEquals(vec, retrieved);
        }
    }

    @Test
    public void testGetIntoBufferMethods() {
        try (Index index = new Index.Config().metric("cos").dimensions(3).build()) {
            float[] vecF32 = { 1.0f, 2.0f, 3.0f };
            index.reserve(10);
            index.add(42, vecF32);

            float[] bufferF32 = new float[3];
            index.getInto(42, bufferF32);
            assertArrayEquals(vecF32, bufferF32, 0.01f);
        }

        try (Index index = new Index.Config().metric("cos").dimensions(3).quantization("f64").build()) {
            double[] vecF64 = { 1.0, 2.0, 3.0 };
            index.reserve(10);
            index.add(43, vecF64);

            double[] bufferF64 = new double[3];
            index.getInto(43, bufferF64);
            assertArrayEquals(vecF64, bufferF64, 0.01);
        }

        try (Index index = new Index.Config().metric("cos").dimensions(4).quantization("i8").build()) {
            byte[] vecI8 = { 10, 20, 30, 40 };
            index.reserve(10);
            index.add(44, vecI8);

            byte[] bufferI8 = new byte[4];
            index.getInto(44, bufferI8);
            assertArrayEquals(vecI8, bufferI8);
        }
    }

    @Test
    public void testConcurrentAdd() throws Exception {
        try (Index index = new Index.Config().metric("cos").dimensions(4).build()) {
            index.reserve(1000);
            
            ExecutorService executor = Executors.newFixedThreadPool(10);
            CompletableFuture<Void>[] futures = new CompletableFuture[10];
            
            for (int t = 0; t < 10; t++) {
                final int threadId = t;
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = 0; i < 50; i++) {
                        long key = threadId * 50L + i;
                        float[] vector = randomVector(4);
                        index.add(key, vector);
                    }
                }, executor);
            }
            
            CompletableFuture.allOf(futures).get(10, TimeUnit.SECONDS);
            executor.shutdown();
            
            assertEquals(500, index.size());
        }
    }

    @Test
    public void testConcurrentSearch() throws Exception {
        try (Index index = new Index.Config().metric("cos").dimensions(4).build()) {
            index.reserve(100);
            
            // Add some vectors first
            for (int i = 0; i < 100; i++) {
                index.add(i, randomVector(4));
            }
            
            ExecutorService executor = Executors.newFixedThreadPool(5);
            CompletableFuture<long[]>[] futures = new CompletableFuture[5];
            
            for (int t = 0; t < 5; t++) {
                futures[t] = CompletableFuture.supplyAsync(() -> {
                    float[] queryVector = randomVector(4);
                    return index.search(queryVector, 10);
                }, executor);
            }
            
            for (CompletableFuture<long[]> future : futures) {
                long[] results = future.get(10, TimeUnit.SECONDS);
                assertTrue(results.length > 0);
                assertTrue(results.length <= 10);
            }
            
            executor.shutdown();
        }
    }

    @Test
    public void testMixedConcurrency() throws Exception {
        try (Index index = new Index.Config().metric("cos").dimensions(3).build()) {
            index.reserve(200);
            
            ExecutorService executor = Executors.newFixedThreadPool(8);
            CompletableFuture<Void>[] addFutures = new CompletableFuture[4];
            CompletableFuture<Void>[] searchFutures = new CompletableFuture[4];
            
            // Add operations
            for (int t = 0; t < 4; t++) {
                final int threadId = t;
                addFutures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = 0; i < 30; i++) {
                        long key = threadId * 30L + i;
                        index.add(key, randomVector(3));
                    }
                }, executor);
            }
            
            // Wait for some adds to complete, then start searches
            Thread.sleep(100);
            
            // Search operations
            for (int t = 0; t < 4; t++) {
                searchFutures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = 0; i < 10; i++) {
                        float[] queryVector = randomVector(3);
                        long[] results = index.search(queryVector, 5);
                        assertTrue(results.length >= 0);
                    }
                }, executor);
            }
            
            CompletableFuture.allOf(addFutures).get(15, TimeUnit.SECONDS);
            CompletableFuture.allOf(searchFutures).get(15, TimeUnit.SECONDS);
            executor.shutdown();
            
            assertEquals(120, index.size());
        }
    }

    @Test
    public void testBatchAdd() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            index.reserve(10);
            
            // Create a batch of 3 vectors concatenated
            float[] batchVectors = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            index.add(100, batchVectors);  // Should add keys 100, 101, 102
            
            assertEquals(3, index.size());
            
            // Verify each vector was added correctly
            assertArrayEquals(new float[]{1.0f, 2.0f}, index.get(100), 0.01f);
            assertArrayEquals(new float[]{3.0f, 4.0f}, index.get(101), 0.01f);
            assertArrayEquals(new float[]{5.0f, 6.0f}, index.get(102), 0.01f);
        }
    }

    @Test
    public void testBatchAddDouble() {
        try (Index index = new Index.Config().metric("cos").dimensions(3).quantization("f64").build()) {
            index.reserve(10);
            
            // Create a batch of 2 double vectors concatenated
            double[] batchVectors = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            index.add(200, batchVectors);  // Should add keys 200, 201
            
            assertEquals(2, index.size());
            
            double[] buffer = new double[3];
            index.getInto(200, buffer);
            assertArrayEquals(new double[]{1.0, 2.0, 3.0}, buffer, 0.01);
            
            index.getInto(201, buffer);
            assertArrayEquals(new double[]{4.0, 5.0, 6.0}, buffer, 0.01);
        }
    }

    @Test
    public void testBatchAddByte() {
        try (Index index = new Index.Config().metric("cos").dimensions(4).quantization("i8").build()) {
            index.reserve(10);
            
            // Create a batch of 2 byte vectors concatenated
            byte[] batchVectors = {10, 20, 30, 40, 50, 60, 70, 80};
            index.add(300, batchVectors);  // Should add keys 300, 301
            
            assertEquals(2, index.size());
            
            byte[] buffer = new byte[4];
            index.getInto(300, buffer);
            assertArrayEquals(new byte[]{10, 20, 30, 40}, buffer);
            
            index.getInto(301, buffer);
            assertArrayEquals(new byte[]{50, 60, 70, 80}, buffer);
        }
    }

    @Test
    public void testBatchDetection() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            index.reserve(10);
            
            // Valid batch: 4 elements = 2 vectors of dimension 2
            float[] validBatch = {1.0f, 2.0f, 3.0f, 4.0f};
            index.add(10, validBatch);
            assertEquals(2, index.size());
            
            // Invalid batch: 3 elements, not divisible by dimensions (2)
            float[] invalidBatch = {1.0f, 2.0f, 3.0f};
            assertThrows(IllegalArgumentException.class, () -> index.add(20, invalidBatch));
            
            // Still should be 2 vectors (invalid batch should not have been added)
            assertEquals(2, index.size());
        }
    }
}
