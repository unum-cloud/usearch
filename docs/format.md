# USearch Format Specification

> ⚠️ **Note**: The format is expected to change in the 3.0 release to support sparse indexes and offer even faster startup times when `view`-ing large indexes from disk.

## Table of Contents

- [Dense Indexes](#dense-indexes)
- [Core](#core)
- [Fragmented Storage](#fragmented-storage)

## Dense Indexes

Dense index files consist of two main parts: [a binary matrix of all vectors](#matrix-blob) and the [index](#index-blob) itself.

### Matrix BLOB

The matrix BLOB is optionally prepended to the index file. It starts with the number of `rows` and `columns`, which can be either 32-bit or 64-bit unsigned integers.

- The number of `rows` corresponds to the number of vectors in the index.
- The number of `columns` corresponds to the number of bytes in a single vector.

### Index BLOB

The index BLOB is composed of three main sections:

1. **Metadata**: Occupies 64 bytes and contains the following fields:
    1. 7-byte magic string `usearch` to infer the MIME type of the file.
    2. 3-byte version number, containing the `major`, `minor`, and `patch` release numbers.
    3. 1-byte enums for `metric`, `scalar`, `key`, and `compressed_slot` types.
    4. 8-byte integers for the number of present vectors, number of deleted vectors, and number of dimensions.
    5. 1-byte flags for multi-vector support.

2. **Levels**: Contains 1-byte integers representing the level each node occupies. This information is used to estimate memory requirements and to enable memory-mapping of the entire file.

3. **Core**: Detailed in the [Core section](#core). This format is consistent across all types of indexes.

## Fragmented Storage

The matrix BLOB can be stored separately and is not required to successfully open the file.

## Core

The core section of the index is serialized as a sequence of nodes, each representing a point in the index. This section immediately follows the `levels` block in the serialized format.

### Header

The core section begins with a header containing the following fields:

- `uint64_t size`: Number of nodes in the index.
- `uint64_t connectivity`: Maximum number of edges for a node.
- `uint64_t connectivity_base`: Base connectivity value.
- `uint64_t max_level`: Maximum level in the multi-level graph.
- `uint64_t entry_slot`: Entry slot for the graph.

These fields are serialized in a contiguous block of bytes, in the order listed.

### Levels

Following the header, the `levels` block contains a sequence of 1-byte integers. Each integer represents the level of a node in the index. The number of levels is equal to the `size` field in the header.

### Nodes

After the `levels` block, the nodes are serialized. Each node is represented as a contiguous block of bytes. The size and format of these blocks are implementation-specific. Nodes are serialized in the same order as their corresponding levels in the `levels` block.
