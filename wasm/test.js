var assert = require('assert');
var module = require('./usearch-wasm.js');

var usearch = new module.UsearchWasm('usearch_wasm.wasm');

var index = usearch.init({ metricKind: "usearch_metric_ip_k", quantization: "scalar-f32-k", dimensions: 2, connectivity: 16, expansionAdd: 40, expansionSearch: 16 });
assert.equal(index.tag, "ok", "tag should be ok")

var result = usearch.connectivity(index, error);
assert.equal(result.tag, "ok", "tag should be ok")
assert.equal(result.val, 16n, "connectivity should be 16");

// assert.equal(usearch.dimensions(index, error), 2n, 'dimensions should be 2');
// assert.equal(usearch.size(index, error), 0n, 'initial size should be 0');

console.log(`JavaScript WASM test passed!`);
