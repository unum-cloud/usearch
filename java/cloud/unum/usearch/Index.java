package cloud.unum.usearch;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.io.InputStream;

/**
 * @brief Java bindings for Unum USearch.
 * 
 * @see Tutorials:
 *      https://nachtimwald.com/2017/06/06/wrapping-a-c-library-in-java/
 *      https://www3.ntu.edu.sg/home/ehchua/programming/java/javanativeinterface.html
 */
public class Index {

  private long c_ptr = 0;

  public Index(String metric, String accuracy, //
      long dimensions, long capacity, long connectivity, //
      long expansion_add, long expansion_search) {
    c_ptr = c_create(
        metric,
        accuracy,
        dimensions,
        capacity,
        connectivity,
        expansion_add,
        expansion_search);
  }

  public long size() {
    return c_size(c_ptr);
  }

  public long connectivity() {
    return c_connectivity(c_ptr);
  }

  public long dimensions() {
    return c_dimensions(c_ptr);
  }

  public long capacity() {
    return c_capacity(c_ptr);
  }

  public void reserve(long capacity) {
    c_reserve(c_ptr, capacity);
  }

  public void add(int label, float vector[]) {
    c_add(c_ptr, label, vector);
  }

  public int[] search(float vector[], long count) {
    return c_search(c_ptr, vector, count);
  }

  public void save(String path) {
    c_save(c_ptr, path);
  }

  public void load(String path) {
    c_load(c_ptr, path);
  }

  public void view(String path) {
    c_view(c_ptr, path);
  }

  public static class Config {
    private String _metric = "ip";
    private String _accuracy = "f32";
    private long _dimensions = 0;
    private long _capacity = 0;
    private long _connectivity = 0;
    private long _expansion_add = 0;
    private long _expansion_search = 0;

    public Config() {
    }

    public Index build() {
      return new Index(
          _metric,
          _accuracy,
          _dimensions,
          _capacity,
          _connectivity,
          _expansion_add,
          _expansion_search);
    }

    public Config metric(String _metric) {
      this._metric = _metric;
      return this;
    }

    public Config accuracy(String _accuracy) {
      this._accuracy = _accuracy;
      return this;
    }

    public Config dimensions(long _dimensions) {
      this._dimensions = _dimensions;
      return this;
    }

    public Config capacity(long _capacity) {
      this._capacity = _capacity;
      return this;
    }

    public Config connectivity(long _connectivity) {
      this._connectivity = _connectivity;
      return this;
    }

    public Config expansion_add(long _expansion_add) {
      this._expansion_add = _expansion_add;
      return this;
    }

    public Config expansion_search(long _expansion_search) {
      this._expansion_search = _expansion_search;
      return this;
    }
  }

  static {
    System.loadLibrary("usearch");
  }

  public static void main(String[] args) {
    Index index = new Index.Config().metric("cos").dimensions(100).build();
    index.size();
  }

  private static native long c_create(//
      String metric, String accuracy, //
      long dimensions, long capacity, long connectivity, //
      long expansion_add, long expansion_search);

  private static native void c_destroy(long ptr);

  private static native long c_size(long ptr);

  private static native long c_connectivity(long ptr);

  private static native long c_dimensions(long ptr);

  private static native long c_capacity(long ptr);

  private static native void c_reserve(long ptr, long capacity);

  private static native void c_add(long ptr, int label, float vector[]);

  private static native int[] c_search(long ptr, float vector[], long count);

  private static native void c_save(long ptr, String path);

  private static native void c_load(long ptr, String path);

  private static native void c_view(long ptr, String path);
}