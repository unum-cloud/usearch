import struct

import numpy as np


def load_matrix(
        filename: str,
        start_row: int = 0, count_rows: int = None,
        view: bool = False) -> np.ndarray:
    """Read *.ibin, *.bbib, *.hbin, *.fbin, *.dbin files with matrices.

    :param filename: path to the matrix file
    :param start_row: start reading vectors from this index
    :param count_rows: number of vectors to read. If None, read all vectors
    :param view: set to `True` to memory-map the file instead of loading to RAM

    :return: parsed matrix
    :rtype: numpy.ndarray
    """
    dtype = np.float32
    scalar_size = 4
    if filename.endswith('.fbin'):
        dtype = np.float32
        scalar_size = 4
    elif filename.endswith('.dbin'):
        dtype = np.float64
        scalar_size = 8
    elif filename.endswith('.hbin'):
        dtype = np.float16
        scalar_size = 2
    elif filename.endswith('.ibin'):
        dtype = np.int32
        scalar_size = 4
    elif filename.endswith('.bbin'):
        dtype = np.int8
        scalar_size = 1
    else:
        raise Exception('Unknown file type')

    with open(filename, 'rb') as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32)
        rows = (rows - start_row) if count_rows is None else count_rows
        row_offset = start_row * scalar_size * cols

        if view:
            return np.memmap(
                f, dtype=dtype, mode='r', offset=8+row_offset, shape=(rows, cols))
        else:
            return np.fromfile(
                f, count=rows * cols,
                dtype=dtype, offset=row_offset).reshape(rows, cols)


def save_matrix(vectors: np.ndarray, filename: str):
    """Write *.ibin, *.bbib, *.hbin, *.fbin, *.dbin files with matrices.

    :param vectors: the matrix to serialize
    :type vectors: numpy.ndarray
    :param filename: path to the matrix file
    :type filename: str
    """
    dtype = np.float32
    if filename.endswith('.fbin'):
        dtype = np.float32
    elif filename.endswith('.dbin'):
        dtype = np.float64
    elif filename.endswith('.hbin'):
        dtype = np.float16
    elif filename.endswith('.ibin'):
        dtype = np.int32
    elif filename.endswith('.bbin'):
        dtype = np.int8
    else:
        raise Exception('Unknown file type')

    assert len(vectors.shape) == 2, 'Input array must have 2 dimensions'
    with open(filename, 'wb') as f:
        count, dim = vectors.shape
        f.write(struct.pack('<i', count))
        f.write(struct.pack('<i', dim))
        vectors.astype(dtype).flatten().tofile(f)
