/**
 * Java bindings for Unum USearch vector search library.
 * 
 * <p>USearch is a high-performance approximate nearest neighbor (ANN) search engine
 * optimized for vector similarity search. This Java binding provides a convenient
 * interface to the underlying C++ implementation.
 * 
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Multiple distance metrics (Cosine, Squared Euclidean, Inner Product, Haversine, etc.)</li>
 *   <li>Multiple quantization types (Float64, Float32, BFloat16, Float16, Int8, Binary)</li>
 *   <li>SIMD-accelerated distance calculations</li>
 *   <li>Memory-efficient storage with configurable precision</li>
 *   <li>Thread-safe operations for concurrent construction or search</li>
 *   <li>Persistent storage with save/load capabilities</li>
 * </ul>
 * 
 * <h2>Basic Usage:</h2>
 * <pre>{@code
 * // Create an index for 128-dimensional vectors using cosine similarity
 * try (Index index = new Index.Config()
 *         .metric(Index.Metric.COSINE)
 *         .quantization(Index.Quantization.FLOAT32)
 *         .dimensions(128)
 *         .connectivity(16)
 *         .build()) {
 *     
 *     // Add vectors
 *     float[] vector1 = {1.0f, 2.0f, 3.0f, ...}; // 128 dimensions
 *     index.add(42L, vector1);
 *     
 *     // Search for similar vectors
 *     float[] query = {1.1f, 2.1f, 3.1f, ...};
 *     long[] results = index.search(query, 10); // Find 10 nearest neighbors
 *     
 *     // Retrieve a vector by key
 *     float[] retrieved = index.get(42L);
 * }
 * }</pre>
 * 
 * <h2>Advanced Configuration:</h2>
 * <pre>{@code
 * Index index = new Index.Config()
 *     .metric(Index.Metric.EUCLIDEAN_SQUARED)    // Distance metric (L2² - no sqrt)
 *     .quantization(Index.Quantization.FLOAT16)  // Storage precision  
 *     .dimensions(768)                           // Vector dimensions
 *     .capacity(1000000)                         // Expected number of vectors
 *     .connectivity(32)                          // Graph connectivity (higher = better recall)
 *     .expansion_add(128)                        // Search width during insertion
 *     .expansion_search(64)                      // Search width during queries
 *     .build();
 * }</pre>
 * 
 * <h2>Thread Safety:</h2>
 * <p>USearch index operations are thread-safe for many concurrent reads or many concurrent writes.
 * Operations except {@code search()} and {@code add()} shouldn't be called concurrently.</p>
 * 
 * <h2>Memory Management:</h2>
 * <p>This class implements {@link AutoCloseable} for automatic resource management.
 * Always use try-with-resources or explicitly call {@link #close()} to free native memory.</p>
 * 
 * @see <a href="https://github.com/unum-cloud/usearch">USearch GitHub Repository</a>
 * @see <a href="https://unum-cloud.github.io/usearch/">USearch Documentation</a>
 */
package cloud.unum.usearch;

import java.io.IOException;

public class Index implements AutoCloseable {

  /**
   * Distance metric constants for vector similarity calculations.
   * These constants can be used with {@link Config#metric(String)}.
   */
  public static final class Metric {
    /** Inner product (dot product) similarity */
    public static final String INNER_PRODUCT = "ip";
    /** Cosine similarity for normalized vectors */
    public static final String COSINE = "cos";
    /** Squared Euclidean distance (L2²) - faster than true Euclidean, no sqrt */
    public static final String EUCLIDEAN_SQUARED = "l2sq";
    /** Haversine distance for geographic coordinates */
    public static final String HAVERSINE = "haversine";
    /** Hamming distance for binary vectors */
    public static final String HAMMING = "hamming";
    /** Jaccard similarity coefficient */
    public static final String JACCARD = "jaccard";
  }

  /**
   * Scalar quantization types for vector storage.
   * These constants can be used with {@link Config#quantization(String)}.
   */
  public static final class Quantization {
    /** 64-bit floating point (double precision) */
    public static final String FLOAT64 = "f64";
    /** 32-bit floating point (single precision) */
    public static final String FLOAT32 = "f32";
    /** Brain Float 16 (half precision with a wider exponent) */
    public static final String BFLOAT16 = "bf16";
    /** 16-bit floating point (half precision) */
    public static final String FLOAT16 = "f16";
    /** 8-bit integer quantization */
    public static final String INT8 = "i8";
    /** Binary quantization (1 bit per dimension, 8 dimensions per word) */
    public static final String BINARY = "b1";
  }

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
    this(c_create(metric, quantization, dimensions, capacity, connectivity, expansion_add, expansion_search));
  }

  private Index(long c_ptr) {
    this.c_ptr = c_ptr;
  }

  /**
   * Loads an index from a file into memory.
   *
   * @param path path to load from
   * @return a mutable Index.
   * @throws Error if any part of loading from path failed.
   */
  public static Index loadFromPath(String path) {
    return new Index(c_createFromFile(path, false));
  }

  /**
   * Loads an index view from a file into memory.
   *
   * @param path path to load from
   * @return an immutable Index.
   * @throws Error if any part of loading from path failed.
   */
  public static Index viewFromPath(String path) {
    return new Index(c_createFromFile(path, true));
  }

  @Override
  public void close() {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_destroy(c_ptr);
    c_ptr = 0;
  }

  /**
   * Retrieves the current size of the index.
   *
   * @return the number of vectors currently indexed.
   */
  public long size() {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_size(c_ptr);
  }

  /**
   * Retrieves the connectivity parameter of the index.
   *
   * @return the connectivity parameter that limits connections-per-node in graph.
   */
  public long connectivity() {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_connectivity(c_ptr);
  }

  /**
   * Retrieves the number of dimensions of the vectors in the index.
   *
   * @return the number of dimensions in the vectors.
   */
  public long dimensions() {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_dimensions(c_ptr);
  }

  /**
   * Retrieves the current capacity of the index.
   *
   * @return the total capacity including current size.
   */
  public long capacity() {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_capacity(c_ptr);
  }

  /**
   * Reserves memory for a specified number of incoming vectors.
   *
   * @param capacity the desired total capacity including current size.
   */
  public void reserve(long capacity) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_reserve(c_ptr, capacity);
  }

  /**
   * Adds a vector with a specified key to the index.
   *
   * @param key    the key associated with the vector
   * @param vector the vector data
   */
  public void add(long key, float vector[]) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_add_f32(c_ptr, key, vector);
  }

  /**
   * Searches for closest vectors to the specified query vector.
   *
   * @param vector the query vector data
   * @param count  the number of nearest neighbors to search
   * @return an array of keys of the nearest neighbors
   */
  public long[] search(float vector[], long count) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_search_f32(c_ptr, vector, count);
  }

  /**
   * Return the contents of the vector at key.
   *
   * @param key key to lookup.
   * @return the contents of the vector.
   * @throws IllegalArgumentException is key is not available.
   */
  public float[] get(long key) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_get(c_ptr, key);
  }

  /**
   * Adds a double precision vector with a specified key to the index.
   *
   * @param key    the key associated with the vector
   * @param vector the double precision vector data
   */
  public void add(long key, double vector[]) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_add_f64(c_ptr, key, vector);
  }

  /**
   * Searches for closest vectors to the specified double precision query vector.
   *
   * @param vector the double precision query vector data
   * @param count  the number of nearest neighbors to search
   * @return an array of keys of the nearest neighbors
   */
  public long[] search(double vector[], long count) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_search_f64(c_ptr, vector, count);
  }

  /**
   * Adds an int8 quantized vector with a specified key to the index.
   *
   * @param key    the key associated with the vector
   * @param vector the int8 quantized vector data
   */
  public void add(long key, byte vector[]) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_add_i8(c_ptr, key, vector);
  }

  /**
   * Searches for closest vectors to the specified int8 quantized query vector.
   *
   * @param vector the int8 quantized query vector data
   * @param count  the number of nearest neighbors to search
   * @return an array of keys of the nearest neighbors
   */
  public long[] search(byte vector[], long count) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_search_i8(c_ptr, vector, count);
  }

  /**
   * Retrieves the vector at the specified key and populates the provided float
   * buffer.
   *
   * @param key    key to lookup.
   * @param buffer buffer to populate with vector data.
   * @throws IllegalArgumentException if key is not available or buffer size is
   *                                  incorrect.
   */
  public void getInto(long key, float[] buffer) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_get_into_f32(c_ptr, key, buffer);
  }

  /**
   * Retrieves the vector at the specified key and populates the provided double
   * buffer.
   *
   * @param key    key to lookup.
   * @param buffer buffer to populate with vector data.
   * @throws IllegalArgumentException if key is not available or buffer size is
   *                                  incorrect.
   */
  public void getInto(long key, double[] buffer) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_get_into_f64(c_ptr, key, buffer);
  }

  /**
   * Retrieves the vector at the specified key and populates the provided byte
   * buffer.
   *
   * @param key    key to lookup.
   * @param buffer buffer to populate with vector data.
   * @throws IllegalArgumentException if key is not available or buffer size is
   *                                  incorrect.
   */
  public void getInto(long key, byte[] buffer) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_get_into_i8(c_ptr, key, buffer);
  }

  /**
   * Saves the index to a file.
   *
   * @param path the file path where the index will be saved.
   */
  public void save(String path) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_save(c_ptr, path);
  }

  /**
   * Loads the index from a file.
   *
   * @param path the file path from where the index will be loaded.
   */
  public void load(String path) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_load(c_ptr, path);
  }

  /**
   * Creates a view of the index from a file without copying it into memory.
   *
   * @param path the file path from where the view will be created.
   */
  public void view(String path) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    c_view(c_ptr, path);
  }

  /**
   * Removes the vector associated with the given key from the index.
   *
   * @param key the key of the vector to be removed.
   * @return {@code true} if the vector was successfully removed, {@code false}
   *         otherwise.
   */
  public boolean remove(long key) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
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
  public boolean rename(long from, long to) {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_rename(c_ptr, from, to);
  }

  /**
   * Retrieves the memory usage of the index in bytes.
   * This includes both the graph structure and stored vectors.
   *
   * @return the total memory usage in bytes
   */
  public long memoryUsage() {
    if (c_ptr == 0) {
      throw new IllegalStateException("Index already closed");
    }
    return c_memory_usage(c_ptr);
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
        loadLibraryFromJar();
      } catch (IOException e1) {
        throw new RuntimeException("Failed to load USearch native library: " + e1.getMessage(), e1);
      }
    }
  }

  private static void loadLibraryFromJar() throws IOException {
    String osName = System.getProperty("os.name").toLowerCase();
    String osArch = System.getProperty("os.arch").toLowerCase();

    String libName;
    if (osName.contains("mac") || osName.contains("darwin")) {
      libName = "libusearch_jni.dylib";
    } else if (osName.contains("windows")) {
      libName = "libusearch_jni.dll";
    } else {
      libName = "libusearch_jni.so";
    }

    // Try architecture-specific first, then fall back to generic
    String[] searchPaths = {
        "/usearch-native/" + getArchSpecificPath() + "/" + libName, // e.g., /usearch-native/linux-x86_64/libusearch.so
        "/usearch-native/" + libName // fallback to generic path
    };

    IOException lastException = null;
    for (String path : searchPaths) {
      try {
        NativeUtils.loadLibraryFromJar(path);
        return; // Success!
      } catch (IOException e) {
        lastException = e;
        // Continue to next path
      }
    }

    throw new IOException("Could not find native library for " + osName + " " + osArch +
        ". Tried paths: " + String.join(", ", searchPaths), lastException);
  }

  private static String getArchSpecificPath() {
    String osName = System.getProperty("os.name").toLowerCase();
    String osArch = System.getProperty("os.arch").toLowerCase();

    // Normalize architecture names
    String normalizedArch;
    if (osArch.equals("amd64") || osArch.equals("x86_64")) {
      normalizedArch = "amd64";
    } else if (osArch.equals("aarch64") || osArch.equals("arm64")) {
      normalizedArch = "arm64";
    } else if (osArch.equals("x86") || osArch.equals("i386")) {
      normalizedArch = "x86";
    } else if (osArch.equals("armv7l") || osArch.contains("armv7")) {
      normalizedArch = "arm32";
    } else {
      normalizedArch = osArch;
    }

    // Detect Android vs regular Linux
    boolean isAndroid = System.getProperty("java.vendor", "").toLowerCase().contains("android") ||
        System.getProperty("java.vm.name", "").toLowerCase().contains("dalvik") ||
        System.getProperty("java.specification.vendor", "").toLowerCase().contains("android");

    // Create platform-specific path
    if (osName.contains("mac") || osName.contains("darwin")) {
      return "darwin-" + normalizedArch;
    } else if (osName.contains("windows")) {
      return "windows-" + normalizedArch;
    } else if (isAndroid) {
      return "android-" + normalizedArch;
    } else {
      return "linux-" + normalizedArch;
    }
  }

  /**
   * A simple main method to test the Index functionalities.
   *
   * @param args command line arguments (not used in this case)
   */
  public static void main(String[] args) {
    try (
        Index index = new Index.Config().metric("cos").dimensions(100).build()) {
      index.size();
    }
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

  private static native long c_createFromFile(String path, boolean view);

  private static native void c_destroy(long ptr);

  private static native long c_size(long ptr);

  private static native long c_connectivity(long ptr);

  private static native long c_dimensions(long ptr);

  private static native long c_capacity(long ptr);

  private static native void c_reserve(long ptr, long capacity);

  private static native void c_save(long ptr, String path);

  private static native void c_load(long ptr, String path);

  private static native void c_view(long ptr, String path);

  private static native boolean c_remove(long ptr, long key);

  private static native boolean c_rename(long ptr, long from, long to);

  private static native float[] c_get(long ptr, long key);

  // Overloaded methods:
  private static native void c_add_f32(long ptr, long key, float vector[]);

  private static native void c_add_f64(long ptr, long key, double vector[]);

  private static native void c_add_i8(long ptr, long key, byte vector[]);

  private static native long[] c_search_f32(long ptr, float vector[], long count);

  private static native long[] c_search_f64(long ptr, double vector[], long count);

  private static native long[] c_search_i8(long ptr, byte vector[], long count);

  private static native void c_get_into_f32(long ptr, long key, float buffer[]);

  private static native void c_get_into_f64(long ptr, long key, double buffer[]);

  private static native void c_get_into_i8(long ptr, long key, byte buffer[]);

}
