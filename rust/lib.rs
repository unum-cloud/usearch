#[cxx::bridge]
pub mod ffi {

    // Shared structs with fields visible to both languages.
    struct Matches {
        keys: Vec<u64>,
        distances: Vec<f32>,
    }

    enum MetricKind {
        IP,
        L2sq,
        Cos,
        Pearson,
        Haversine,
        Hamming,
        Tanimoto,
        Sorensen,
    }

    enum ScalarKind {
        F64,
        F32,
        F16,
        I8,
        B1,
    }

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

        /// Low-level C++ interface, that is further wrapped into the high-level `Index`
        type NativeIndex;

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
        pub fn save_to_buffer(self: &NativeIndex, buffer: &mut [u8]) -> Result<()>;
        pub fn load_from_buffer(self: &NativeIndex, buffer: &[u8]) -> Result<()>;
        pub fn view_from_buffer(self: &NativeIndex, buffer: &[u8]) -> Result<()>;
    }
}

pub struct Index {
    inner: cxx::UniquePtr<ffi::NativeIndex>,
}

impl Default for ffi::IndexOptions {
    fn default() -> Self {
        use crate::ffi::MetricKind;
        use crate::ffi::ScalarKind;
        Self {
            dimensions: 128,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            connectivity: 32,
            expansion_add: 2,
            expansion_search: 3,
            multi: false,
        }
    }
}

pub trait VectorType {
    fn add(index: &Index, key: u64, vector: &[Self]) -> Result<(), cxx::Exception>
    where
        Self: Sized;
    fn get(index: &Index, key: u64, buffer: &mut [Self]) -> Result<usize, cxx::Exception>
    where
        Self: Sized;
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized;
}

impl VectorType for i8 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_i8(query, count)
    }
    fn get(index: &Index, key: u64, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_i8(key, vector)
    }
    fn add(index: &Index, key: u64, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_i8(key, vector)
    }
}

impl VectorType for f32 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f32(query, count)
    }
    fn get(index: &Index, key: u64, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f32(key, vector)
    }
    fn add(index: &Index, key: u64, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f32(key, vector)
    }
}

impl VectorType for f64 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f64(query, count)
    }
    fn get(index: &Index, key: u64, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f64(key, vector)
    }
    fn add(index: &Index, key: u64, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f64(key, vector)
    }
}

impl Index {
    pub fn new(options: &ffi::IndexOptions) -> Result<Self, cxx::Exception> {
        match ffi::new_native_index(options) {
            Ok(inner) => Result::Ok(Self { inner }),
            Err(err) => Err(err),
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

    /// Adds a vector with a specified key to the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn add<T: VectorType>(self: &Index, key: u64, vector: &[T]) -> Result<(), cxx::Exception> {
        T::add(self, key, vector)
    }

    /// Extracts one or more vectors matching specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn get<T: VectorType>(
        self: &Index,
        key: u64,
        vector: &mut [T],
    ) -> Result<usize, cxx::Exception> {
        T::get(self, key, vector)
    }

    /// Reserves memory for a specified number of incoming vectors.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The desired total capacity including the current size.
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
    pub fn remove(self: &Index, key: u64) -> Result<usize, cxx::Exception> {
        self.inner.remove(key)
    }

    /// Renames the vector under a certain key.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the vector to be renamed.
    /// * `to` - The new name.
    ///
    /// # Returns
    ///
    /// `true` if the vector is successfully renamed, `false` otherwise.
    pub fn rename(self: &Index, from: u64, to: u64) -> Result<usize, cxx::Exception> {
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
    pub fn contains(self: &Index, key: u64) -> bool {
        self.inner.contains(key)
    }

    /// Count the count of vector with the same specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// Number of vectors found.
    pub fn count(self: &Index, key: u64) -> usize {
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

    use crate::new_index;
    use crate::Index;

    #[test]
    fn integration() {
        let mut options = IndexOptions::default();
        options.dimensions = 5;

        let index = Index::new(&options).unwrap();

        assert!(index.reserve(10).is_ok());
        assert!(index.capacity() >= 10);
        assert!(index.connectivity() != 0);
        assert_eq!(index.dimensions(), 5);
        assert_eq!(index.size(), 0);

        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];

        assert!(index.add(42, &first).is_ok());
        assert!(index.add(43, &second).is_ok());
        assert_eq!(index.size(), 2);

        // Read back the tags
        let results = index.search(&first, 10).unwrap();
        assert_eq!(results.keys.len(), 2);

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
    }
}
