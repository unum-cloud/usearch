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
        new Index();
        System.out.println("Java Tests Passed!");
    }
}
