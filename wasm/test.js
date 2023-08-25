var assert = require('assert');
var module = require('./usearch-wasm.js');

var error = ""
var usearch = new module.UsearchWasm('usearch.wasm', error);

var index = usearch.init({ metricKind: "usearch_metric_ip_k", quantization: "scalar-f32-k", dimensions: 2, connectivity: 16, expansionAdd: 40, expansionSearch: 16 });
assert.equal(usearch.connectivity(index, error), 16n, 'connectivity should be 16');
assert.equal(usearch.dimensions(index, error), 2n, 'dimensions should be 2');
assert.equal(usearch.size(index, error), 0n, 'initial size should be 0');

console.log(`JavaScript WASM test passed!`);
