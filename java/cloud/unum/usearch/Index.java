
/**
 * Java bindings for Unum USearch vector search library.
 */
package cloud.unum.usearch;

import java.io.IOException;

/**
 * Java bindings for Unum USearch.
 * <p>
 * Provides interface to interact with USearch library and perform various
 * operations related to indexing and searching.
 *
 * @see <a href=
 *      "https://nachtimwald.com/2017/06/06/wrapping-a-c-library-in-java/">Wrapping
 *      a C library in Java</a>
 * @see <a href=
 *      "https://www3.ntu.edu.sg/home/ehchua/programming/java/javanativeinterface.html">Java
 *      Native Interface tutorial</a>
 */
public class Index {

  private long c_ptr = 0;

  /**
   * Creates a new instance of Index with specified parameters.
   *
   * @param metric           the metric type for distance calculation between
   *                         vectors
   * @param quantization     the scalar type for quantization of vector data
   * @param dimensions       the number of dimensions in the vectors
   * @param capacity         the initial capacity of the index
   * @param connectivity     the connectivity parameter that limits
   *                         connections-per-node in graph
   * @param expansion_add    the expansion factor used for index construction when
   *                         adding vectors
   * @param expansion_search the expansion factor used for index construction
   *                         during search operations
   */
  public Index(
      String metric,
      String quantization,
      long dimensions,
      long capacity,
      long connectivity,
      long expansion_add,
      long expansion_search) {
    c_ptr = c_create(
        metric,
        quantization,
        dimensions,
        capacity,
        connectivity,
        expansion_add,
        expansion_search);
  }

  /**
   * Retrieves the current size of the index.
   *
   * @return the number of vectors currently indexed.
   */
  public long size() {
    return c_size(c_ptr);
  }

  /**
   * Retrieves the connectivity parameter of the index.
   *
   * @return the connectivity parameter that limits connections-per-node in graph.
   */
  public long connectivity() {
    return c_connectivity(c_ptr);
  }

  /**
   * Retrieves the number of dimensions of the vectors in the index.
   *
   * @return the number of dimensions in the vectors.
   */
  public long dimensions() {
    return c_dimensions(c_ptr);
  }

  /**
   * Retrieves the current capacity of the index.
   *
   * @return the total capacity including current size.
   */
  public long capacity() {
    return c_capacity(c_ptr);
  }

  /**
   * Reserves memory for a specified number of incoming vectors.
   *
   * @param capacity the desired total capacity including current size.
   */
  public void reserve(long capacity) {
    c_reserve(c_ptr, capacity);
  }

  /**
   * Adds a vector with a specified key to the index.
   *
   * @param key    the key associated with the vector
   * @param vector the vector data
   */
  public void add(int key, float vector[]) {
    c_add(c_ptr, key, vector);
  }

  /**
   * Searches for closest vectors to the specified query vector.
   *
   * @param vector the query vector data
   * @param count  the number of nearest neighbors to search
   * @return an array of keys of the nearest neighbors
   */
  public int[] search(float vector[], long count) {
    return c_search(c_ptr, vector, count);
  }

  /**
   * Saves the index to a file.
   *
   * @param path the file path where the index will be saved.
   */
  public void save(String path) {
    c_save(c_ptr, path);
  }

  /**
   * Loads the index from a file.
   *
   * @param path the file path from where the index will be loaded.
   */
  public void load(String path) {
    c_load(c_ptr, path);
  }

  /**
   * Creates a view of the index from a file without copying it into memory.
   *
   * @param path the file path from where the view will be created.
   */
  public void view(String path) {
    c_view(c_ptr, path);
  }

  /**
   * Removes the vector associated with the given key from the index.
   *
   * @param key the key of the vector to be removed.
   * @return {@code true} if the vector was successfully removed, {@code false}
   *         otherwise.
   */
  public boolean remove(int key) {
    return c_remove(c_ptr, key);
  }

  /**
   * Renames the vector to map to a different key.
   *
   * @param from the key of the vector to be renamed.
   * @param to   the new key for the vector.
   * @return {@code true} if the vector was successfully renamed, {@code false}
   *         otherwise.
   */
  public boolean rename(int from, int to) {
    return c_rename(c_ptr, from, to);
  }

  /**
   * Configuration class for building an Index instance.
   * <p>
   * This class provides a builder pattern to set various configurations for an
   * Index. Once all configurations
   * are set, calling {@link #build()} will produce an instance of {@link Index}.
   */
  public static class Config {

    private String _metric = "ip";
    private String _quantization = "f32";
    private long _dimensions = 0;
    private long _capacity = 0;
    private long _connectivity = 0;
    private long _expansion_add = 0;
    private long _expansion_search = 0;

    /**
     * Default constructor for the Config class.
     */
    public Config() {
    }

    /**
     * Constructs an Index instance based on the current configuration settings.
     *
     * @return a newly constructed Index instance.
     */
    public Index build() {
      return new Index(
          _metric,
          _quantization,
          _dimensions,
          _capacity,
          _connectivity,
          _expansion_add,
          _expansion_search);
    }

    /**
     * Sets the metric for distance calculation between vectors.
     *
     * @param _metric the metric type
     * @return this configuration instance
     */
    public Config metric(String _metric) {
      this._metric = _metric;
      return this;
    }

    /**
     * Sets the scalar type for quantization of vector data.
     *
     * @param _quantization the quantization type
     * @return this configuration instance
     */
    public Config quantization(String _quantization) {
      this._quantization = _quantization;
      return this;
    }

    /**
     * Sets the number of dimensions in the vectors.
     *
     * @param _dimensions the number of dimensions
     * @return this configuration instance
     */
    public Config dimensions(long _dimensions) {
      this._dimensions = _dimensions;
      return this;
    }

    /**
     * Sets the initial capacity of the index.
     *
     * @param _capacity the index capacity
     * @return this configuration instance
     */
    public Config capacity(long _capacity) {
      this._capacity = _capacity;
      return this;
    }

    /**
     * Sets the connectivity parameter that limits connections-per-node in the
     * graph.
     *
     * @param _connectivity the connectivity value
     * @return this configuration instance
     */
    public Config connectivity(long _connectivity) {
      this._connectivity = _connectivity;
      return this;
    }

    /**
     * Sets the expansion factor used for index construction when adding vectors.
     *
     * @param _expansion_add the expansion factor for adding vectors
     * @return this configuration instance
     */
    public Config expansion_add(long _expansion_add) {
      this._expansion_add = _expansion_add;
      return this;
    }

    /**
     * Sets the expansion factor used for index construction during search
     * operations.
     *
     * @param _expansion_search the expansion factor for search
     * @return this configuration instance
     */
    public Config expansion_search(long _expansion_search) {
      this._expansion_search = _expansion_search;
      return this;
    }
  }

  static {
    try {
      System.loadLibrary("usearch"); // used for tests. This library in classpath only
    } catch (UnsatisfiedLinkError e) {
      try {
        NativeUtils.loadLibraryFromJar("/usearch/libusearch.so");
      } catch (IOException e1) {
        throw new RuntimeException(e1);
      }
    }
  }

  /**
   * A simple main method to test the Index functionalities.
   *
   * @param args command line arguments (not used in this case)
   */
  public static void main(String[] args) {
    Index index = new Index.Config().metric("cos").dimensions(100).build();
    index.size();
    System.out.println("Java tests passed!");
  }

  private static native long c_create(
      String metric,
      String quantization,
      long dimensions,
      long capacity,
      long connectivity,
      long expansion_add,
      long expansion_search);

  private static native void c_destroy(long ptr);

  private static native long c_size(long ptr);

  private static native long c_connectivity(long ptr);

  private static native long c_dimensions(long ptr);

  private static native long c_capacity(long ptr);

  private static native void c_reserve(long ptr, long capacity);

  private static native void c_add(long ptr, int key, float vector[]);

  private static native int[] c_search(long ptr, float vector[], long count);

  private static native void c_save(long ptr, String path);

  private static native void c_load(long ptr, String path);

  private static native void c_view(long ptr, String path);

  private static native boolean c_remove(long ptr, int key);

  private static native boolean c_rename(long ptr, int from, int to);
}
