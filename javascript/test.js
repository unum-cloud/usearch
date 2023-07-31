var assert = require('assert');
var usearch = require('bindings')('usearch');

var index = new usearch.Index({ metric: 'l2sq', connectivity: 16, dimensions: 2 })
assert.equal(index.connectivity(), 16)
assert.equal(index.dimensions(), 2)
assert.equal(index.size(), 0)

index.add(15, new Float32Array([10, 20]))
index.add(16, new Float32Array([10, 25]))
assert.equal(index.size(), 2)

var results = index.search(new Float32Array([13, 14]), 2)
assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]))
assert.deepEqual(results.distances, new Float32Array([45, 130]))

// Batch

var index2 = new usearch.Index({ metric: 'l2sq', connectivity: 16, dimensions: 2 })

const keys = [15, 16]
const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])]
index2.add(keys, vectors)
assert.equal(index.size(), 2)

var results = index.search(new Float32Array([13, 14]), 2)
assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]))
assert.deepEqual(results.distances, new Float32Array([45, 130]))

console.log('JavaScript tests passed!')