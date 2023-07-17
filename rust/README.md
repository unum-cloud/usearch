# USearch for Rust

## Installation

```sh
cargo add usearch
```

## Quickstart

```rust

let options = IndexOptions {
            dimensions: 5,
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
assert_eq!(results.count, 2);
```

## Multi-Threading

```rust
assert!(index.add_in_thread(42, &first, 0).is_ok());
assert!(index.add_in_thread(43, &second, 0).is_ok());
let results = index.search_in_thread(&first, 10, 0).unwrap();
```

Being a systems-programming language, Rust has better control over memory management and concurrency but lacks function overloading.
Aside from the `add` and `search`, USearch Rust binding also provides `add_in_thread` and `search_in_thread`, which let users identify the calling thread to use underlying temporary memory more efficiently.

## Serialization

```rust
assert!(index.save("index.usearch").is_ok());
assert!(index.load("index.usearch").is_ok());
assert!(index.view("index.usearch").is_ok());
```

## Metrics

```rust
assert!(new_l2sq(3, &quant, 0, 0, 0).is_ok());
assert!(new_cos(3, &quant, 0, 0, 0).is_ok());
assert!(new_haversine(&quant, 0, 0, 0).is_ok());
```
