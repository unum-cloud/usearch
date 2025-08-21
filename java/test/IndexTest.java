import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

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
}
