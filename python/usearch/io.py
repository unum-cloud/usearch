import struct
import numpy as np


def load_matrix(filename: str, start_row: int = 0, count_rows: int = None, view: bool = False):
    """
    Read *.ibin, *.hbin, *.fbin, *.dbin files with matrixes.
    Args:
        :param filename (str): path to the matrix file
        :param start_row (int): start reading vectors from this index
        :param count_rows (int): number of vectors to read. If None, read all vectors
    Returns:
        Parsed matrix (numpy.ndarray)
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
    else:
        raise Exception('Unknown file type')

    with open(filename, 'rb') as f:
        rows, cols = np.fromfile(f, count=2, dtype=np.int32)
        rows = (rows - start_row) if count_rows is None else count_rows
        row_offset = start_row * scalar_size * cols

        if view:
            return np.memmap(f, dtype=dtype, mode='r', offset=8+row_offset, shape=(rows, cols))
        else:
            return np.fromfile(f, count=rows * cols, dtype=dtype, offset=row_offset).reshape(rows, cols)


def save_matrix(vecs: np.array, filename: str):
    """
    Write *.ibin, *.hbin, *.fbin, *.dbin files with matrixes.
    Args:
        :param vecs (numpy.array): the matrix to serialize
        :param filename (str): path to the matrix file
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
    else:
        raise Exception('Unknown file type')

    assert len(vecs.shape) == 2, 'Input array must have 2 dimensions'
    with open(filename, 'wb') as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype(dtype).flatten().tofile(f)
