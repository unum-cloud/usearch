import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.io.File;
import java.io.IOException;

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
            int[] keys = index.search(vec, 5);
        }

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
}
