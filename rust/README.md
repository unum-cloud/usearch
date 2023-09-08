# USearch for Rust

## Installation

```sh
cargo add usearch
```

You can find [interface documentation on docs.rs](https://docs.rs/usearch/2.3.0/usearch/ffi/struct.Index.html).

## Quickstart

```rust

let options = IndexOptions {
    dimensions: 3,
    metric: MetricKind::IP,
    quantization: ScalarKind::F16,
    connectivity: 0,
    expansion_add: 0,
    expansion_search: 0
};

let index = new_index(&options).unwrap();

assert!(index.reserve(10).is_ok());
assert!(index.capacity() >= 10);
assert!(index.connectivity() != 0);
assert_eq!(index.dimensions(), 3);
assert_eq!(index.size(), 0);

let first: [f32; 3] = [0.2, 0.1, 0.2];
let second: [f32; 3] = [0.2, 0.1, 0.2];

assert!(index.add(42, &first).is_ok());
assert!(index.add(43, &second).is_ok());
assert_eq!(index.size(), 2);

// Read back the tags
let results = index.search(&first, 10).unwrap();
assert_eq!(results.keys.len(), 2);
```

## Serialization

```rust
assert!(index.save("index.usearch").is_ok());
assert!(index.load("index.usearch").is_ok());
assert!(index.view("index.usearch").is_ok());
```

Similarly, serializing to/from buffers is supported.

```rust
assert!(index.save_to_buffer(&mut serialization_buffer).is_ok());
assert!(index.load_from_buffer(&serialization_buffer).is_ok());
assert!(index.view_from_buffer(&serialization_buffer).is_ok());
```

## Metrics

```rust
assert!(new_l2sq(3, &quant, 0, 0, 0).is_ok());
assert!(new_cos(3, &quant, 0, 0, 0).is_ok());
assert!(new_haversine(&quant, 0, 0, 0).is_ok());
```
