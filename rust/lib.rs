use std::boxed::Box;

/// The key type used to identify vectors in the index.
/// It is a 64-bit unsigned integer.
pub type Key = u64;

/// The distance type used to represent the similarity between vectors.
/// It is a 32-bit floating-point number.
pub type Distance = f32;

/// Callback signature for custom metric functions, defined in the Rust layer and used in the C++ layer.
pub type StatefullMetric = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
) -> Distance;

/// Callback signature for custom predicate functions, defined in the Rust layer and used in the C++ layer.
pub type StatefullPredicate = unsafe extern "C" fn(Key, *mut std::ffi::c_void) -> bool;

#[cxx::bridge]
pub mod ffi {

    /// The metric kind used to differentiate built-in distance functions.
    #[derive(Debug)]
    #[repr(i32)]
    enum MetricKind {
        Unknown,
        /// The Inner Product metric, defined as `IP = 1 - sum(a[i] * b[i])`.
        IP,
        /// The squared Euclidean Distance metric, defined as `L2 = sum((a[i] - b[i])^2)`.
        L2sq,
        /// The Cosine Similarity metric, defined as `Cos = 1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2) * sqrt(sum(b[i]^2)))`.
        Cos,
        /// The Pearson Correlation metric.
        Pearson,
        /// The Haversine (Great Circle) Distance metric.
        Haversine,
        /// The Jensen Shannon Divergence metric.
        Divergence,
        /// The bit-level Hamming Distance metric, defined as the number of differing bits.
        Hamming,
        /// The bit-level Tanimoto (Jaccard) metric, defined as the number of intersecting bits divided by the number of union bits.
        Tanimoto,
        /// The bit-level Sorensen metric.
        Sorensen,
    }

    /// The scalar kind used to differentiate built-in vector element types.
    #[derive(Debug)]
    #[repr(i32)]
    enum ScalarKind {
        Unknown,
        /// 64-bit double-precision IEEE 754 floating-point number.
        F64,
        /// 32-bit single-precision IEEE 754 floating-point number.
        F32,
        /// 16-bit half-precision IEEE 754 floating-point number (different from `bf16`).
        F16,
        /// 8-bit signed integer.
        I8,
        /// 1-bit binary value, packed 8 per byte.
        B1,
    }

    /// The resulting matches from a search operation.
    /// It contains the keys and distances of the closest vectors.
    #[derive(Debug)]
    struct Matches {
        keys: Vec<u64>,
        distances: Vec<f32>,
    }

    /// The index options used to configure the dense index during creation.
    /// It contains the number of dimensions, the metric kind, the scalar kind,
    /// the connectivity, the expansion values, and the multi-flag.
    #[derive(Debug, PartialEq)]
    struct IndexOptions {
        dimensions: usize,
        metric: MetricKind,
        quantization: ScalarKind,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        multi: bool,
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("lib.hpp");

        /// Low-level C++ interface that is further wrapped into the high-level `Index`
        type NativeIndex;

        pub fn expansion_add(self: &NativeIndex) -> usize;
        pub fn expansion_search(self: &NativeIndex) -> usize;
        pub fn change_expansion_add(self: &NativeIndex, n: usize);
        pub fn change_expansion_search(self: &NativeIndex, n: usize);
        pub fn change_metric_kind(self: &NativeIndex, metric: MetricKind);

        /// Changes the metric function used to calculate the distance between vectors.
        /// Avoids the `std::ffi::c_void` type and the `StatefullMetric` type, that the FFI
        /// does not support, replacing them with basic pointer-sized integer types.
        /// The first two arguments are the pointers to the vectors to compare, and the third
        /// argument is the `metric_state` propagated from the Rust layer.
        pub fn change_metric(self: &NativeIndex, metric: usize, metric_state: usize);

        pub fn new_native_index(options: &IndexOptions) -> Result<UniquePtr<NativeIndex>>;
        pub fn reserve(self: &NativeIndex, capacity: usize) -> Result<()>;
        pub fn dimensions(self: &NativeIndex) -> usize;
        pub fn connectivity(self: &NativeIndex) -> usize;
        pub fn size(self: &NativeIndex) -> usize;
        pub fn capacity(self: &NativeIndex) -> usize;
        pub fn serialized_length(self: &NativeIndex) -> usize;

        pub fn add_i8(self: &NativeIndex, key: u64, vector: &[i8]) -> Result<()>;
        pub fn add_f16(self: &NativeIndex, key: u64, vector: &[u16]) -> Result<()>;
        pub fn add_f32(self: &NativeIndex, key: u64, vector: &[f32]) -> Result<()>;
        pub fn add_f64(self: &NativeIndex, key: u64, vector: &[f64]) -> Result<()>;

        pub fn search_i8(self: &NativeIndex, query: &[i8], count: usize) -> Result<Matches>;
        pub fn search_f16(self: &NativeIndex, query: &[u16], count: usize) -> Result<Matches>;
        pub fn search_f32(self: &NativeIndex, query: &[f32], count: usize) -> Result<Matches>;
        pub fn search_f64(self: &NativeIndex, query: &[f64], count: usize) -> Result<Matches>;

        pub fn filtered_search_i8(
            self: &NativeIndex,
            query: &[i8],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_f16(
            self: &NativeIndex,
            query: &[u16],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_f32(
            self: &NativeIndex,
            query: &[f32],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_f64(
            self: &NativeIndex,
            query: &[f64],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;

        pub fn get_i8(self: &NativeIndex, key: u64, buffer: &mut [i8]) -> Result<usize>;
        pub fn get_f16(self: &NativeIndex, key: u64, buffer: &mut [u16]) -> Result<usize>;
        pub fn get_f32(self: &NativeIndex, key: u64, buffer: &mut [f32]) -> Result<usize>;
        pub fn get_f64(self: &NativeIndex, key: u64, buffer: &mut [f64]) -> Result<usize>;

        pub fn remove(self: &NativeIndex, key: u64) -> Result<usize>;
        pub fn rename(self: &NativeIndex, from: u64, to: u64) -> Result<usize>;
        pub fn contains(self: &NativeIndex, key: u64) -> bool;
        pub fn count(self: &NativeIndex, key: u64) -> usize;

        pub fn save(self: &NativeIndex, path: &str) -> Result<()>;
        pub fn load(self: &NativeIndex, path: &str) -> Result<()>;
        pub fn view(self: &NativeIndex, path: &str) -> Result<()>;
        pub fn reset(self: &NativeIndex) -> Result<()>;
        pub fn memory_usage(self: &NativeIndex) -> usize;
        pub fn hardware_acceleration(self: &NativeIndex) -> *const c_char;

        pub fn save_to_buffer(self: &NativeIndex, buffer: &mut [u8]) -> Result<()>;
        pub fn load_from_buffer(self: &NativeIndex, buffer: &[u8]) -> Result<()>;
        pub fn view_from_buffer(self: &NativeIndex, buffer: &[u8]) -> Result<()>;
    }
}

pub enum MetricFunction {
    I8Metric(Box<dyn Fn(*const i8, *const i8) -> Distance + Send + Sync>),
    F16Metric(Box<dyn Fn(*const u16, *const u16) -> Distance + Send + Sync>),
    F32Metric(Box<dyn Fn(*const f32, *const f32) -> Distance + Send + Sync>),
    F64Metric(Box<dyn Fn(*const f64, *const f64) -> Distance + Send + Sync>),
}

/// Approximate Nearest Neighbors search index for dense vectors.
///
/// The `Index` struct provides an abstraction over a dense vector space, allowing
/// for efficient addition, search, and management of high-dimensional vectors.
/// It supports various distance metrics and vector types through generic interfaces.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
///
/// // Create an index with specific options
/// let mut options = IndexOptions::default();
/// options.dimensions = 256; // Set the number of dimensions for vectors
/// options.metric = MetricKind::Cos; // Use cosine similarity for distance measurement
/// options.quantization = ScalarKind::F32; // Use 32-bit floating point numbers
///
/// let index = Index::new(&options).expect("Failed to create index.");
///
/// // Add vectors to the index
/// let vector1: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, ...];
/// let vector2: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, ...];
/// index.add(1, &vector1).expect("Failed to add vector1.");
/// index.add(2, &vector2).expect("Failed to add vector2.");
///
/// // Search for the nearest neighbors to a query vector
/// let query: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5, ...];
/// let results = index.search(&query, 5).expect("Search failed.");
/// for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
///     println!("Key: {}, Distance: {}", key, distance);
/// }
/// ```
///
pub struct Index {
    inner: cxx::UniquePtr<ffi::NativeIndex>,
    metric_fn: Option<MetricFunction>,
}

impl Default for ffi::IndexOptions {
    fn default() -> Self {
        use crate::ffi::MetricKind;
        use crate::ffi::ScalarKind;
        Self {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        }
    }
}

impl Clone for ffi::IndexOptions {
    fn clone(&self) -> Self {
        ffi::IndexOptions {
            dimensions: (self.dimensions),
            metric: (self.metric),
            quantization: (self.quantization),
            connectivity: (self.connectivity),
            expansion_add: (self.expansion_add),
            expansion_search: (self.expansion_search),
            multi: (self.multi),
        }
    }
}

/// The `VectorType` trait defines operations for managing and querying vectors
/// in an index. It supports generic operations on vectors of different types,
/// allowing for the addition, retrieval, and search of vectors within an index.
pub trait VectorType {
    /// Adds a vector to the index under the specified key.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the vector is to be added.
    /// - `key`: The key under which the vector should be stored.
    /// - `vector`: A slice representing the vector to be added.
    ///
    /// # Returns
    /// - `Ok(())` if the vector was successfully added to the index.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception>
    where
        Self: Sized;

    /// Retrieves a vector from the index by its key.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` from which the vector is to be retrieved.
    /// - `key`: The key of the vector to retrieve.
    /// - `buffer`: A mutable slice where the retrieved vector will be stored. The size of the
    ///   buffer determines the maximum number of elements that can be retrieved.
    ///
    /// # Returns
    /// - `Ok(usize)` indicating the number of elements actually written into the `buffer`.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    fn get(index: &Index, key: Key, buffer: &mut [Self]) -> Result<usize, cxx::Exception>
    where
        Self: Sized;

    /// Performs a search in the index using the given query vector, returning
    /// up to `count` closest matches.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the search is to be performed.
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches found.
    /// - `Err(cxx::Exception)` if an error occurred during the search operation.
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized;

    /// Performs a filtered search in the index using a query vector and a custom
    /// filter function, returning up to `count` matches that satisfy the filter.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the search is to be performed.
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    /// - `filter`: A closure that takes a `Key` and returns `true` if the corresponding
    ///   vector should be included in the search results, or `false` otherwise.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches that satisfy the filter.
    /// - `Err(cxx::Exception)` if an error occurred during the filtered search operation.
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool;

    /// Changes the metric used for distance calculations within the index.
    ///
    /// # Parameters
    /// - `index`: A mutable reference to the `Index` for which the metric is to be changed.
    /// - `metric`: A boxed closure that defines the new metric for distance calculation. The
    ///   closure must take two pointers to elements of type `Self` and return a `Distance`.
    ///
    /// # Returns
    /// - `Ok(())` if the metric was successfully changed.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    fn change_metric(
        index: &mut Index,
        metric: Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception>
    where
        Self: Sized;
}

impl VectorType for f32 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f32(query, count)
    }
    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f32(key, vector)
    }
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f32(key, vector)
    }
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline::<F> as *const ());
            let closure_address: usize = &filter as *const F as usize;
            index
                .inner
                .filtered_search_f32(query, count, trampoline_fn, closure_address)
        }
    }

    fn change_metric(
        index: &mut Index,
        metric: Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = fn(*const f32, *const f32) -> Distance;
        index.metric_fn = Some(MetricFunction::F32Metric(metric));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const f32;
            let second_ptr = second as *const f32;
            let closure: MetricFn = unsafe { std::mem::transmute(closure_address) };
            closure(first_ptr, second_ptr)
        }

        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline as *const ());
            let closure_address = match index.metric_fn {
                Some(MetricFunction::F32Metric(ref metric)) => metric as *const _ as usize,
                _ => panic!("Expected F32Metric"),
            };
            index.inner.change_metric(trampoline_fn, closure_address)
        }

        Ok(())
    }
}

impl VectorType for i8 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_i8(query, count)
    }
    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_i8(key, vector)
    }
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_i8(key, vector)
    }
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline::<F> as *const ());
            let closure_address: usize = &filter as *const F as usize;
            index
                .inner
                .filtered_search_i8(query, count, trampoline_fn, closure_address)
        }
    }
    fn change_metric(
        index: &mut Index,
        metric: Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = fn(*const i8, *const i8) -> Distance;
        index.metric_fn = Some(MetricFunction::I8Metric(metric));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const i8;
            let second_ptr = second as *const i8;
            let closure: MetricFn = unsafe { std::mem::transmute(closure_address) };
            closure(first_ptr, second_ptr)
        }

        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline as *const ());
            let closure_address = match index.metric_fn {
                Some(MetricFunction::I8Metric(ref metric)) => metric as *const _ as usize,
                _ => panic!("Expected I8Metric"),
            };
            index.inner.change_metric(trampoline_fn, closure_address)
        }

        Ok(())
    }
}

impl VectorType for f64 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f64(query, count)
    }
    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f64(key, vector)
    }
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f64(key, vector)
    }
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline::<F> as *const ());
            let closure_address: usize = &filter as *const F as usize;
            index
                .inner
                .filtered_search_f64(query, count, trampoline_fn, closure_address)
        }
    }
    fn change_metric(
        index: &mut Index,
        metric: Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = fn(*const f64, *const f64) -> Distance;
        index.metric_fn = Some(MetricFunction::F64Metric(metric));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const f64;
            let second_ptr = second as *const f64;
            let closure: MetricFn = unsafe { std::mem::transmute(closure_address) };
            closure(first_ptr, second_ptr)
        }

        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline as *const ());
            let closure_address = match index.metric_fn {
                Some(MetricFunction::F64Metric(ref metric)) => metric as *const _ as usize,
                _ => panic!("Expected F64Metric"),
            };
            index.inner.change_metric(trampoline_fn, closure_address)
        }

        Ok(())
    }
}

impl Index {
    pub fn new(options: &ffi::IndexOptions) -> Result<Self, cxx::Exception> {
        match ffi::new_native_index(options) {
            Ok(inner) => Result::Ok(Self {
                inner,
                metric_fn: None,
            }),
            Err(err) => Err(err),
        }
    }

    /// Retrieves the expansion value used during index creation.
    pub fn expansion_add(self: &Index) -> usize {
        self.inner.expansion_add()
    }

    /// Retrieves the expansion value used during search.
    pub fn expansion_search(self: &Index) -> usize {
        self.inner.expansion_search()
    }

    /// Updates the expansion value used during index creation. Rarely used.
    pub fn change_expansion_add(self: &Index, n: usize) {
        self.inner.change_expansion_add(n)
    }

    /// Updates the expansion value used during search operations.
    pub fn change_expansion_search(self: &Index, n: usize) {
        self.inner.change_expansion_search(n)
    }

    /// Changes the metric kind used to calculate the distance between vectors.
    pub fn change_metric_kind(self: &Index, metric: ffi::MetricKind) {
        self.inner.change_metric_kind(metric)
    }

    /// Overrides the metric function used to calculate the distance between vectors.
    pub fn change_metric<T: VectorType>(
        self: &mut Index,
        metric: Box<dyn Fn(*const T, *const T) -> Distance + Send + Sync>,
    ) {
        T::change_metric(self, metric).unwrap();
    }

    /// Retrieves the hardware acceleration information.
    pub fn hardware_acceleration(&self) -> String {
        use core::ffi::CStr;
        unsafe {
            let c_str = CStr::from_ptr(self.inner.hardware_acceleration());
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to the provided query.
    ///
    /// # Arguments
    ///
    /// * `query` - A slice containing the query vector data.
    /// * `count` - The maximum number of neighbors to search for.
    ///
    /// # Returns
    ///
    /// A `Result` containing the matches found.
    pub fn search<T: VectorType>(
        self: &Index,
        query: &[T],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        T::search(self, query, count)
    }

    /// Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to the provided query
    /// satisfying a custom filter function.
    ///
    /// # Arguments
    ///
    /// * `query` - A slice containing the query vector data.
    /// * `count` - The maximum number of neighbors to search for.
    /// * `filter` - A closure that takes a `Key` and returns `true` if the corresponding vector should be included in the search results, or `false` otherwise.
    ///
    /// # Returns
    ///
    /// A `Result` containing the matches found.
    pub fn filtered_search<T: VectorType, F>(
        self: &Index,
        query: &[T],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        F: Fn(Key) -> bool,
    {
        T::filtered_search(self, query, count, filter)
    }

    /// Adds a vector with a specified key to the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn add<T: VectorType>(self: &Index, key: Key, vector: &[T]) -> Result<(), cxx::Exception> {
        T::add(self, key, vector)
    }

    /// Extracts one or more vectors matching the specified key.
    /// The `vector` slice must be a multiple of the number of dimensions in the index.
    /// After the execution, return the number `X` of vectors found.
    /// The vector slice's first `X * dimensions` elements will be filled.
    ///
    /// If you are a novice user, consider `export`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn get<T: VectorType>(
        self: &Index,
        key: Key,
        vector: &mut [T],
    ) -> Result<usize, cxx::Exception> {
        T::get(self, key, vector)
    }

    /// Extracts one or more vectors matching specified key into supplied resizable vector.
    /// The `vector` is resized to a multiple of the number of dimensions in the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A mutable vector containing the vector data.
    pub fn export<T: VectorType + Default + Clone>(
        self: &Index,
        key: Key,
        vector: &mut Vec<T>,
    ) -> Result<usize, cxx::Exception> {
        let dim = self.dimensions();
        let max_matches = self.count(key);
        vector.resize(dim * max_matches, T::default());
        let matches = T::get(self, key, &mut vector[..]);
        if matches.is_err() {
            return matches;
        }
        vector.resize(dim * matches.as_ref().unwrap(), T::default());
        return matches;
    }

    /// Reserves memory for a specified number of incoming vectors.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The desired total capacity, including the current size.
    pub fn reserve(self: &Index, capacity: usize) -> Result<(), cxx::Exception> {
        self.inner.reserve(capacity)
    }

    /// Retrieves the number of dimensions in the vectors indexed.
    pub fn dimensions(self: &Index) -> usize {
        self.inner.dimensions()
    }

    /// Retrieves the connectivity parameter that limits connections-per-node in the graph.
    pub fn connectivity(self: &Index) -> usize {
        self.inner.connectivity()
    }

    /// Retrieves the current number of vectors in the index.
    pub fn size(self: &Index) -> usize {
        self.inner.size()
    }

    /// Retrieves the total capacity of the index, including reserved space.
    pub fn capacity(self: &Index) -> usize {
        self.inner.capacity()
    }

    /// Reports expected file size after serialization.
    pub fn serialized_length(self: &Index) -> usize {
        self.inner.serialized_length()
    }

    /// Removes the vector associated with the given key from the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the vector to be removed.
    ///
    /// # Returns
    ///
    /// `true` if the vector is successfully removed, `false` otherwise.
    pub fn remove(self: &Index, key: Key) -> Result<usize, cxx::Exception> {
        self.inner.remove(key)
    }

    /// Renames the vector under a specific key.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the vector to be renamed.
    /// * `to` - The new name.
    ///
    /// # Returns
    ///
    /// `true` if the vector is renamed, `false` otherwise.
    pub fn rename(self: &Index, from: Key, to: Key) -> Result<usize, cxx::Exception> {
        self.inner.rename(from, to)
    }

    /// Checks if the index contains a vector with a specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// `true` if the index contains the vector with the given key, `false` otherwise.
    pub fn contains(self: &Index, key: Key) -> bool {
        self.inner.contains(key)
    }

    /// Count the count of vectors with the same specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// Number of vectors found.
    pub fn count(self: &Index, key: Key) -> usize {
        self.inner.count(key)
    }

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the index will be saved.
    pub fn save(self: &Index, path: &str) -> Result<(), cxx::Exception> {
        self.inner.save(path)
    }

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the index will be loaded.
    pub fn load(self: &Index, path: &str) -> Result<(), cxx::Exception> {
        self.inner.load(path)
    }

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the view will be created.
    pub fn view(self: &Index, path: &str) -> Result<(), cxx::Exception> {
        self.inner.view(path)
    }

    /// Erases all members from the index, closes files, and returns RAM to OS.
    pub fn reset(self: &Index) -> Result<(), cxx::Exception> {
        self.inner.reset()
    }

    /// A relatively accurate lower bound on the amount of memory consumed by the system.
    /// In practice, its error will be below 10%.
    pub fn memory_usage(self: &Index) -> usize {
        self.inner.memory_usage()
    }

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the index will be saved.
    pub fn save_to_buffer(self: &Index, buffer: &mut [u8]) -> Result<(), cxx::Exception> {
        self.inner.save_to_buffer(buffer)
    }

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the index will be loaded.
    pub fn load_from_buffer(self: &Index, buffer: &[u8]) -> Result<(), cxx::Exception> {
        self.inner.load_from_buffer(buffer)
    }

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the view will be created.
    pub fn view_from_buffer(self: &Index, buffer: &[u8]) -> Result<(), cxx::Exception> {
        self.inner.view_from_buffer(buffer)
    }
}

pub fn new_index(options: &ffi::IndexOptions) -> Result<Index, cxx::Exception> {
    Index::new(options)
}

#[cfg(test)]
mod tests {
    use crate::ffi::IndexOptions;
    use crate::ffi::MetricKind;
    use crate::ffi::ScalarKind;

    use crate::new_index;
    use crate::Distance;
    use crate::Index;
    use crate::Key;

    use std::env;

    #[test]
    fn print_specs() {
        print!("--------------------------------------------------\n");
        println!("OS: {}", env::consts::OS);
        println!(
            "Rust version: {}",
            env::var("RUST_VERSION").unwrap_or_else(|_| "unknown".into())
        );

        // Create indexes with different configurations
        let f64_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F64,
            ..Default::default()
        })
        .unwrap();

        let f32_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        })
        .unwrap();

        let f16_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16,
            ..Default::default()
        })
        .unwrap();

        let i8_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::I8,
            ..Default::default()
        })
        .unwrap();

        let b1_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            ..Default::default()
        })
        .unwrap();

        println!(
            "f64 hardware acceleration: {}",
            f64_index.hardware_acceleration()
        );
        println!(
            "f32 hardware acceleration: {}",
            f32_index.hardware_acceleration()
        );
        println!(
            "f16 hardware acceleration: {}",
            f16_index.hardware_acceleration()
        );
        println!(
            "i8 hardware acceleration: {}",
            i8_index.hardware_acceleration()
        );
        println!(
            "b1 hardware acceleration: {}",
            b1_index.hardware_acceleration()
        );
        print!("--------------------------------------------------\n");
    }

    #[test]
    fn test_add_get_vector() {
        let mut options = IndexOptions::default();
        options.dimensions = 5;
        options.quantization = ScalarKind::F32;
        let index = Index::new(&options).unwrap();
        assert!(index.reserve(10).is_ok());

        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];
        assert!(index.add(1, &first).is_ok());
        assert!(index.add(2, &second).is_ok());
        assert_eq!(index.size(), 2);

        // Test using Vec<T>
        let mut found_vec: Vec<f32> = Vec::new();
        assert_eq!(index.export(1, &mut found_vec).unwrap(), 1);
        assert_eq!(found_vec.len(), 5);
        assert_eq!(found_vec, first.to_vec());

        // Test using slice
        let mut found_slice = [0.0 as f32; 5];
        assert_eq!(index.get(1, &mut found_slice).unwrap(), 1);
        assert_eq!(found_slice, first);

        // Create a slice with incorrect size
        let mut found = [0.0 as f32; 6]; // This isn't a multiple of the index's dimensions.
        let result = index.get(1, &mut found);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_remove_vector() {
        let mut options = IndexOptions::default();
        options.dimensions = 4;
        options.metric = MetricKind::IP;
        options.quantization = ScalarKind::F64;
        options.connectivity = 10;
        options.expansion_add = 128;
        options.expansion_search = 3;
        let index = Index::new(&options).unwrap();
        assert!(index.reserve(10).is_ok());
        assert_eq!(index.capacity(), 10);

        let first: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
        let second: [f32; 4] = [0.3, 0.2, 0.4, 0.0];

        // IDs until 18446744073709551615 should be fine:
        let id1 = 483367403120493160;
        let id2 = 483367403120558696;
        let id3 = 483367403120624232;
        let id4 = 483367403120624233;

        assert!(index.add(id1, &first).is_ok());
        let mut found_slice = [0.0 as f32; 4];
        assert_eq!(index.get(id1, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id1).is_ok());

        assert!(index.add(id2, &second).is_ok());
        let mut found_slice = [0.0 as f32; 4];
        assert_eq!(index.get(id2, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id2).is_ok());

        assert!(index.add(id3, &second).is_ok());
        let mut found_slice = [0.0 as f32; 4];
        assert_eq!(index.get(id3, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id3).is_ok());

        assert!(index.add(id4, &second).is_ok());
        let mut found_slice = [0.0 as f32; 4];
        assert_eq!(index.get(id4, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id4).is_ok());

        assert_eq!(index.size(), 0);
    }

    #[test]
    fn integration() {
        let mut options = IndexOptions::default();
        options.dimensions = 5;

        let index = Index::new(&options).unwrap();

        assert!(index.expansion_add() > 0);
        assert!(index.expansion_search() > 0);

        assert!(index.reserve(10).is_ok());
        assert!(index.capacity() >= 10);
        assert!(index.connectivity() != 0);
        assert_eq!(index.dimensions(), 5);
        assert_eq!(index.size(), 0);

        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];

        print!("--------------------------------------------------\n");
        println!(
            "before add, memory_usage: {} \
            cap: {} \
            ",
            index.memory_usage(),
            index.capacity(),
        );
        index.change_expansion_add(10);
        assert_eq!(index.expansion_add(), 10);
        assert!(index.add(42, &first).is_ok());
        index.change_expansion_add(12);
        assert_eq!(index.expansion_add(), 12);
        assert!(index.add(43, &second).is_ok());
        assert_eq!(index.size(), 2);
        println!(
            "after add, memory_usage: {} \
            cap: {} \
            ",
            index.memory_usage(),
            index.capacity(),
        );

        index.change_expansion_search(10);
        assert_eq!(index.expansion_search(), 10);
        // Read back the tags
        let results = index.search(&first, 10).unwrap();
        println!("{:?}", results);
        assert_eq!(results.keys.len(), 2);

        index.change_expansion_search(12);
        assert_eq!(index.expansion_search(), 12);
        let results = index.search(&first, 10).unwrap();
        println!("{:?}", results);
        assert_eq!(results.keys.len(), 2);
        print!("--------------------------------------------------\n");

        // Validate serialization
        assert!(index.save("index.rust.usearch").is_ok());
        assert!(index.load("index.rust.usearch").is_ok());
        assert!(index.view("index.rust.usearch").is_ok());

        // Make sure every function is called at least once
        assert!(new_index(&options).is_ok());
        options.metric = MetricKind::L2sq;
        assert!(new_index(&options).is_ok());
        options.metric = MetricKind::Cos;
        assert!(new_index(&options).is_ok());
        options.metric = MetricKind::Haversine;
        assert!(new_index(&options).is_ok());

        let mut serialization_buffer = Vec::new();
        serialization_buffer.resize(index.serialized_length(), 0);
        assert!(index.save_to_buffer(&mut serialization_buffer).is_ok());

        let deserialized_index = new_index(&options).unwrap();
        assert!(deserialized_index
            .load_from_buffer(&serialization_buffer)
            .is_ok());
        assert_eq!(index.size(), deserialized_index.size());

        // reset
        assert_ne!(index.memory_usage(), 0);
        assert!(index.reset().is_ok());
        assert_eq!(index.size(), 0);
        assert_eq!(index.memory_usage(), 0);

        // clone
        options.metric = MetricKind::Haversine;
        let mut opts = options.clone();
        assert_eq!(opts.metric, options.metric);
        assert_eq!(opts.quantization, options.quantization);
        assert_eq!(opts, options);
        opts.metric = MetricKind::Cos;
        assert_ne!(opts.metric, options.metric);
        assert!(new_index(&opts).is_ok());
    }

    #[test]
    fn test_search_with_stateless_filter() {
        let mut options = IndexOptions::default();
        options.dimensions = 5;
        let index = Index::new(&options).unwrap();
        index.reserve(10).unwrap();

        // Adding sample vectors to the index
        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];
        index.add(1, &first).unwrap();
        index.add(2, &second).unwrap();

        // Stateless filter: checks if the key is odd
        let stateless_filter = |key: Key| key % 2 == 1;

        let query = vec![0.2, 0.1, 0.2, 0.1, 0.3]; // Example query vector
        let results = index.filtered_search(&query, 10, stateless_filter).unwrap();
        assert!(
            results.keys.iter().all(|&key| key % 2 == 1),
            "All keys must be odd"
        );
    }

    #[test]
    fn test_search_with_stateful_filter() {
        use std::collections::HashSet;

        let mut options = IndexOptions::default();
        options.dimensions = 5;
        let index = Index::new(&options).unwrap();
        index.reserve(10).unwrap();

        // Adding sample vectors to the index
        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        index.add(1, &first).unwrap();
        index.add(2, &first).unwrap();

        let allowed_keys = vec![1, 2, 3].into_iter().collect::<HashSet<Key>>();
        // Clone `allowed_keys` for use in the closure
        let filter_keys = allowed_keys.clone();
        let stateful_filter = move |key: Key| filter_keys.contains(&key);

        let query = vec![0.2, 0.1, 0.2, 0.1, 0.3]; // Example query vector
        let results = index.filtered_search(&query, 10, stateful_filter).unwrap();

        // Use the original `allowed_keys` for assertion
        assert!(
            results.keys.iter().all(|&key| allowed_keys.contains(&key)),
            "All keys must be in the allowed set"
        );
    }

    #[test]
    fn test_change_distance_function() {
        let mut options = IndexOptions::default();
        options.dimensions = 2; // Adjusted for simplicity in creating test vectors
        let mut index = Index::new(&options).unwrap();
        index.reserve(10).unwrap();

        // Adding a simple vector to test the distance function changes
        let vector: [f32; 2] = [1.0, 0.0];
        index.add(1, &vector).unwrap();

        // Stateless distance function: simply returns the difference in the first element
        let stateless_distance =
            Box::new(|a: *const f32, b: *const f32| unsafe { (*a - *b).abs() });
        index.change_metric(stateless_distance);

        // Now changing to a stateful distance function: scales the difference by a factor
        let scale_factor = 2.0;
        let stateful_distance =
            Box::new(move |a: *const f32, b: *const f32| unsafe { (*a - *b).abs() * scale_factor });
        index.change_metric(stateful_distance);
    }
}
