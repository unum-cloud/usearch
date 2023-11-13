import java.io.File;
import java.util.Arrays;

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

        Index index = new Index.Config().metric("cos").dimensions(2).build();
        float vec[] = { 10, 20 };
        index.reserve(10);
        index.add(42, vec);
        int[] keys = index.search(vec, 5);

        System.out.println("Java Tests Passed!");
    }
}
