const test = require('node:test');
const assert = require('node:assert');
const usearch = require('./usearch.js');

test('Single-entry operations', () => {
    const index = new usearch.Index(2, 'l2sq');

    assert.equal(index.connectivity(), 16, 'connectivity should be 16');
    assert.equal(index.dimensions(), 2, 'dimensions should be 2');
    assert.equal(index.size(), 0, 'initial size should be 0');

    index.add(15n, new Float32Array([10, 20]));
    index.add(16n, new Float32Array([10, 25]));

    assert.equal(index.size(), 2, 'size after adding elements should be 2');
    assert.equal(index.contains(15), true, 'entry must be present after insertion');

    const results = index.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]), 'keys should be 15 and 16');
    assert.deepEqual(results.distances, new Float32Array([45, 130]), 'distances should be 45 and 130');
});

test('Batch operations', () => {
    const indexBatch = new usearch.Index(2, 'l2sq');

    const keys = [15n, 16n];
    const vectors = [new Float32Array([10, 20]), new Float32Array([10, 25])];

    indexBatch.add(keys, vectors);
    assert.equal(indexBatch.size(), 2, 'size after adding batch should be 2');

    const results = indexBatch.search(new Float32Array([13, 14]), 2);

    assert.deepEqual(results.keys, new BigUint64Array([15n, 16n]), 'keys should be 15 and 16');
    assert.deepEqual(results.distances, new Float32Array([45, 130]), 'distances should be 45 and 130');
});

test("Expected results", () => {
    var index = new usearch.Index({
        metric: "cos",
        connectivity: 16,
        dimensions: 3,
    });
    index.add(42n, new Float32Array([0.2, 0.6, 0.4]));
    var results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10);

    assert.equal(index.size(), 1);
    assert.deepEqual(results.keys, new BigUint64Array([42n]));

    // When using mixed-precision, we can't expect the resulting value
    // to be exactly 0, but it should be very close to it.
    var actual = Number(results.distances[0]);
    var expected = 0; // The expected value
    var difference = Math.abs(actual - expected);
    assert.ok(difference < 0.01);
});



test('Operations with invalid values', () => {
    const indexBatch = new usearch.Index(2, 'l2sq');

    const keys = [NaN, 16n];
    const vectors = [new Float32Array([10, 30]), new Float32Array([1, 5])];

    try {
        indexBatch.add(keys, vectors);
        throw new Error('indexBatch.add should have thrown an error.');
    } catch (err) {
        assert.equal(err.message, 'All keys must be integers or bigints.');
    }

    try {
        indexBatch.search(NaN, 2);
        throw new Error('indexBatch.search should have thrown an error.');
    } catch (err) {
        assert.equal(err.message, 'Vectors must be a TypedArray or an array of arrays.');
    }
});
