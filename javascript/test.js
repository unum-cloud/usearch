var assert = require('assert');
var usearch = require('bindings')('usearch');

// Single-entry operations

var index = new usearch.Index({ metric: 'l2sq', connectivity: 16n, dimensions: 2n });
assert.equal(index.connectivity(), 16n, 'connectivity should be 16');
assert.equal(index.dimensions(), 2n, 'dimensions should be 2');
assert.equal(index.size(), 0n, 'initial size should be 0');

index.add(15n, new Float32Array([10, 20]));
index.add(16n, new Float32Array([10, 25]));
assert.equal(index.size(), 2n, 'size after adding elements should be 2');

var results = index.search(new Float32Array([13, 14]), 2n);
assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]), 'keys should be 15 and 16');
assert.deepEqual(results.distances, new Float32Array([45, 130]), 'distances should be 45 and 130');

// Batch operations

var indexBatch = new usearch.Index({ metric: 'l2sq', connectivity: 16n, dimensions: 2n });
const keys = [15n, 16n];
const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])];
indexBatch.add(keys, vectors);
assert.equal(indexBatch.size(), 2, 'size after adding batch should be 2');

results = indexBatch.search(new Float32Array([13, 14]), 2n);
assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]), 'keys should be 15 and 16');
assert.deepEqual(results.distances, new Float32Array([45, 130]), 'distances should be 45 and 130');

console.log('JavaScript tests passed!');
