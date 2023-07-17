# USearch for JavaScript

## Installation

```sh
npm install usearch
```

## Quickstart

```js
var index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 3 })
index.add(42, new Float32Array([0.2, 0.6, 0.4]))
var results = index.search(new Float32Array([0.2, 0.6, 0.4]), 10)

assert.equal(index.size(), 1)
assert.deepEqual(results.labels, new BigUint64Array([42n]))
assert.deepEqual(results.distances, new Float32Array([0]))
```

## Serialization

```js
index.save('index.usearch')
index.load('index.usearch')
index.view('index.usearch')
```
