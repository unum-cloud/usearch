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
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("lib.hpp");

        type NativeIndex;

        pub fn native_index(options: &IndexOptions) -> Result<UniquePtr<NativeIndex>>;
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

        pub fn get_i8(self: &NativeIndex, buffer: &mut[i8], count: usize) -> Result<usize>;
        pub fn get_f16(self: &NativeIndex, buffer: &mut[u16], count: usize) -> Result<usize>;
        pub fn get_f32(self: &NativeIndex, buffer: &mut[f32], count: usize) -> Result<usize>;
        pub fn get_f64(self: &NativeIndex, buffer: &mut[f64], count: usize) -> Result<usize>;

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
    inner: ffi::UniquePtr<ffi::Index>,
}

impl Index {
    pub fn new(options: &ffi::IndexOptions) -> Result<Self, &'static str> {
        let inner = ffi::new_index(options)?;
        Ok(Self { inner })
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
    pub fn search<T>(self: &Index, query: &[T], count: usize) -> Result<ffi::Matches, &'static str> {
        // Use type-based dispatch
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
            inner.search_i8(unsafe { std::mem::transmute::<&[T], &[i8]>(query) }, count)
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            inner.search_f32(unsafe { std::mem::transmute::<&[T], &[f32]>(query) }, count)
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            inner.search_f64(unsafe { std::mem::transmute::<&[T], &[f64]>(query) }, count)
        } else {
            Err("Unsupported type")
        }
    }

    /// Adds a vector with a specified key to the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn add<T>(self: &Index, key: u64, vector: &[T]) -> Result<(), &'static str> {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
            inner.add_i8(key, unsafe { std::mem::transmute::<&[T], &[i8]>(vector) })
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            inner.add_f32(key, unsafe { std::mem::transmute::<&[T], &[f32]>(vector) })
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            inner.add_f64(key, unsafe { std::mem::transmute::<&[T], &[f64]>(vector) })
        } else {
            Err("Unsupported type")
        }
    }

    /// Reserves memory for a specified number of incoming vectors.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The desired total capacity including the current size.
    pub fn reserve(self: &Index, capacity: usize) -> Result<()>;

    /// Retrieves the number of dimensions in the vectors indexed.
    pub fn dimensions(self: &Index) -> usize;

    /// Retrieves the connectivity parameter that limits connections-per-node in the graph.
    pub fn connectivity(self: &Index) -> usize;

    /// Retrieves the current number of vectors in the index.
    pub fn size(self: &Index) -> usize;

    /// Retrieves the total capacity of the index, including reserved space.
    pub fn capacity(self: &Index) -> usize;

    /// Reports expected file size after serialization.
    pub fn serialized_length(self: &Index) -> usize;
    
    /// Removes the vector associated with the given key from the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the vector to be removed.
    ///
    /// # Returns
    ///
    /// `true` if the vector is successfully removed, `false` otherwise.
    pub fn remove(self: &Index, key: u64) -> Result<usize>;

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
    pub fn rename(self: &Index, from: u64, to: u64) -> Result<usize>;

    /// Checks if the index contains a vector with a specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// `true` if the index contains the vector with the given key, `false` otherwise.
    pub fn contains(self: &Index, key: u64) -> bool;

    pub fn count(self: &Index, key: u64) -> usize;

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the index will be saved.
    pub fn save(self: &Index, path: &str) -> Result<()>;

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the index will be loaded.
    pub fn load(self: &Index, path: &str) -> Result<()>;

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the view will be created.
    pub fn view(self: &Index, path: &str) -> Result<()>;

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the index will be saved.
    pub fn save_to_buffer(self: &Index, buffer: &mut [u8]) -> Result<()>;

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the index will be loaded.
    pub fn load_from_buffer(self: &Index, buffer: &[u8]) -> Result<()>;

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the view will be created.
    pub fn view_from_buffer(self: &Index, buffer: &[u8]) -> Result<()>;

}


#[cfg(test)]
mod tests {
    use crate::ffi::new_index;
    use crate::ffi::IndexOptions;
    use crate::ffi::MetricKind;
    use crate::ffi::ScalarKind;

    #[test]
    fn integration() {
        let mut options = IndexOptions {
            dimensions: 5,
            metric: MetricKind::IP,
            quantization: ScalarKind::F16,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
        };

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
