#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from ucall.rich_posix import Server
from usearch.index import Index


def serve(
        ndim: int, metric: str = 'ip',
        port: int = 8545, threads: int = 1,
        path: str = 'index.usearch', immutable: bool = False):

    server = Server(port=port)
    index = Index(ndim=ndim, metric=metric)

    if os.path.exists(path):
        if immutable:
            index.view(path)
        else:
            index.load(path)

    @server
    def add_one(label: int, vector: np.ndarray):
        labels = np.array([label], dtype=np.longlong)
        vectors = vector.reshape(vector.shape[0], 1)
        index.add(labels, vectors, copy=True)

    @server
    def add_ascii(label: int, string: str):
        """
        WARNING: A dirty performance hack!
        Assuming the `f8` vectors in our implementations are just integers,
        and generally contain scalars in the [0, 100] range, we can transmit
        them as JSON-embedded strings. The only symbols we must avoid are
        the double-quote '"' (code 22) and backslash '\' (code 60).
        Printable ASCII characters are in [20, 126].
        """
        vector = np.array(string, dtype=np.int8)
        if np.any((vector < 0) | (vector > 100)):
            return add_one(label, vector)
        # Let's map [0, 100] to the range from [23, 123],
        # poking 60 and replacing with the 124.
        vector += 23
        vector[vector == 60] = 124
        index.add_ascii(label, vector)

    @server
    def add_many(labels: np.ndarray, vectors: np.ndarray):
        labels = labels.astype(np.longlong)
        index.add(labels, vectors, threads=threads, copy=True)

    @server
    def search_one(vector: np.ndarray, count: int) -> np.ndarray:
        vectors = vector.reshape(vector.shape[0], 1)
        results = index.search(vectors, 3)
        return results[0][:results[2][0]]

    @server
    def size() -> int:
        return len(index)

    @server
    def ndim() -> int:
        return index.ndim

    @server
    def capacity() -> int:
        return index.capacity()

    @server
    def connectivity() -> int:
        return index.connectivity()

    try:
        server.run()
    except KeyboardInterrupt:
        if not immutable:
            index.save(path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='log server activity')
    parser.add_argument(
        '--ndim', type=int,
        help='dimensionality of the vectors')
    parser.add_argument(
        '--immutable', type=bool, default=False,
        help='the index can not be updated')

    parser.add_argument(
        '--metric', type=str, default='ip', choices=['ip', 'cos', 'l2sq', 'haversine'],
        help='distance function to compare vectors')
    parser.add_argument(
        '-p', '--port', type=int, default=8545,
        help='port to open for client connections')
    parser.add_argument(
        '-j', '--threads', type=int, default=1,
        help='number of CPU threads to use')
    parser.add_argument(
        '--path', type=str, default='index.usearch',
        help='where to store the index')

    args = parser.parse_args()
    assert args.ndim is not None, 'Define the number of dimensions!'
    serve(
        ndim=args.ndim, metric=args.metric,
        threads=args.threads, port=args.port,
        path=args.path, immutable=args.immutable)
