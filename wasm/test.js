const Module = require('./usearch-wasm.js');

var USearch = new Module.UsearchWasm('usearch.wasm', "");

console.log(`USearch`);
