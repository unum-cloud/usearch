import os
import pytest
import numpy as np

from usearch.io import load_matrix, save_matrix
from usearch.eval import recall_members, random_vectors

from usearch.index import (
    Index,
    SparseIndex,
    MetricKind,
    ScalarKind,
    MetricSignature,
    CompiledMetric,
    Matches,
    MetricKindBitwise,
)
from usearch.index import _normalize_dtype
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


dimensions = [3, 97, 256]
batch_sizes = [1, 33]
index_types = [
    ScalarKind.F32,
    ScalarKind.F64,
    ScalarKind.F16,
    ScalarKind.F8,
]
numpy_types = [np.float32, np.float64, np.float16]

connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]
jit_options = [False]
continuous_metrics = [
    MetricKind.Cos,
    MetricKind.L2sq,
]
hash_metrics = [
    MetricKind.Hamming,
    MetricKind.Tanimoto,
    MetricKind.Sorensen,
]


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_fbin_matrix(rows: int, cols: int):
    original = np.random.rand(rows, cols).astype(np.float32)
    save_matrix(original, "tmp.fbin")
    reconstructed = load_matrix("tmp.fbin")
    assert np.allclose(original, reconstructed)
    os.remove("tmp.fbin")


@pytest.mark.parametrize("rows", batch_sizes)
@pytest.mark.parametrize("cols", dimensions)
def test_serializing_ibin_matrix(rows: int, cols: int):
    original = np.random.randint(0, rows + 1, size=(rows, cols)).astype(np.int32)
    save_matrix(original, "tmp.ibin")
    reconstructed = load_matrix("tmp.ibin")
    assert np.allclose(original, reconstructed)
    os.remove("tmp.ibin")


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("metric", continuous_metrics)
@pytest.mark.parametrize("index_type", index_types)
@pytest.mark.parametrize("numpy_type", numpy_types)
@pytest.mark.parametrize("connectivity", connectivity_options)
@pytest.mark.parametrize("jit", jit_options)
def test_index(
    ndim: int,
    metric: MetricKind,
    index_type: str,
    numpy_type: str,
    connectivity: int,
    jit: bool,
):
    index = Index(
        metric=metric,
        ndim=ndim,
        dtype=index_type,
        connectivity=connectivity,
        expansion_add=DEFAULT_EXPANSION_ADD,
        expansion_search=DEFAULT_EXPANSION_SEARCH,
        jit=jit,
    )
    assert index.ndim == ndim
    assert index.connectivity == connectivity

    vector = random_vectors(count=1, ndim=ndim, dtype=numpy_type).flatten()

    index.add(42, vector)

    assert 42 in index
    assert 42 in index.labels
    assert 43 not in index
    assert index[42] is not None
    assert index[43] is None
    assert len(index[42]) == ndim
    if numpy_type != np.byte:
        assert np.allclose(index[42], vector, atol=0.1)

    matches, distances, count = index.search(vector, 10)
    assert len(index) == 1
    assert len(matches) == count
    assert len(distances) == count
    assert count == 1
    assert matches[0] == 42
    assert distances[0] == pytest.approx(0, abs=1e-3)

    index.save("tmp.usearch")
    index.clear()
    assert len(index) == 0

    index.load("tmp.usearch")
    assert len(index) == 1
    assert len(index[42]) == ndim

    index = Index.restore("tmp.usearch")
    assert len(index) == 1
    assert len(index[42]) == ndim

    # Cleanup
    os.remove("tmp.usearch")


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("metric", continuous_metrics)
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("index_type", index_types)
@pytest.mark.parametrize("numpy_type", numpy_types)
def test_index_batch(
    ndim: int, metric: MetricKind, batch_size: int, index_type: str, numpy_type: str
):
    index = Index(ndim=ndim, metric=metric, dtype=index_type)

    labels = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim, dtype=numpy_type)

    index.add(labels, vectors)
    assert len(index) == batch_size
    assert np.allclose(index.get_vectors(labels).astype(numpy_type), vectors, atol=0.1)

    matches: Matches = index.search(vectors, 10)
    assert matches.labels.shape[0] == matches.distances.shape[0]
    assert matches.counts.shape[0] == batch_size
    assert np.all(np.sort(index.labels) == np.sort(labels))

    if _normalize_dtype(numpy_type) == _normalize_dtype(index_type):
        assert recall_members(index, exact=True) == 1

    index.save("tmp.usearch")
    index.clear()
    assert len(index) == 0

    index.load("tmp.usearch")
    assert len(index) == batch_size
    assert len(index[0]) == ndim

    index = Index.restore("tmp.usearch")
    assert len(index) == batch_size
    assert len(index[0]) == ndim

    # Cleanup
    os.remove("tmp.usearch")


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_index_numba(ndim: int, batch_size: int):
    try:
        from numba import cfunc, types, carray
    except ImportError:
        return

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    signature_two_args = types.float32(
        types.CPointer(types.float32),
        types.CPointer(types.float32),
    )
    signature_three_args = types.float32(
        types.CPointer(types.float32),
        types.CPointer(types.float32),
        types.uint64,
    )
    signature_four_args = types.float32(
        types.CPointer(types.float32),
        types.uint64,
        types.CPointer(types.float32),
        types.uint64,
    )

    @cfunc(signature_two_args)
    def python_inner_product_two_args(a, b):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        c = 0.0
        for i in range(ndim):
            c += a_array[i] * b_array[i]
        return 1 - c

    @cfunc(signature_three_args)
    def python_inner_product_three_args(a, b, ndim):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        c = 0.0
        for i in range(ndim):
            c += a_array[i] * b_array[i]
        return 1 - c

    @cfunc(signature_four_args)
    def python_inner_product_four_args(a, a_ndim, b, b_ndim):
        a_array = carray(a, a_ndim)
        b_array = carray(b, b_ndim)
        c = 0.0
        for i in range(a_ndim):
            c += a_array[i] * b_array[i]
        return 1 - c

    functions = [
        python_inner_product_two_args,
        python_inner_product_three_args,
        python_inner_product_four_args,
    ]
    signatures = [
        MetricSignature.ArrayArray,
        MetricSignature.ArrayArraySize,
        MetricSignature.ArraySizeArraySize,
    ]
    for function, signature in zip(functions, signatures):
        metric = CompiledMetric(
            pointer=function.address,
            kind=MetricKind.IP,
            signature=signature,
        )
        index = Index(ndim=ndim, metric=metric)

        labels = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)

        index.add(labels, vectors)
        matches, distances, count = index.search(vectors, 10)
        assert matches.shape[0] == distances.shape[0]
        assert count.shape[0] == batch_size

        assert recall_members(index, exact=True) == 1


@pytest.mark.parametrize("ndim", dimensions[-1:])
@pytest.mark.parametrize("batch_size", batch_sizes[-1:])
def test_index_cppyy(ndim: int, batch_size: int):
    try:
        import cppyy
        import cppyy.ll
    except ImportError:
        return

    cppyy.cppdef(
        """
    float inner_product_two_args(float *a, float *b) {
        float result = 0;
    #pragma unroll
        for (size_t i = 0; i != ndim; ++i)
            result += a[i] * b[i];
        return 1 - result;
    }
    
    float inner_product_three_args(float *a, float *b, size_t n) {
        float result = 0;
        for (size_t i = 0; i != n; ++i)
            result += a[i] * b[i];
        return 1 - result;
    }
    
    float inner_product_four_args(float *a, size_t an, float *b, size_t) {
        float result = 0;
        for (size_t i = 0; i != an; ++i)
            result += a[i] * b[i];
        return 1 - result;
    }
    """.replace(
            "ndim", str(ndim)
        )
    )

    functions = [
        cppyy.gbl.inner_product_two_args,
        cppyy.gbl.inner_product_three_args,
        cppyy.gbl.inner_product_four_args,
    ]
    signatures = [
        MetricSignature.ArrayArray,
        MetricSignature.ArrayArraySize,
        MetricSignature.ArraySizeArraySize,
    ]
    for function, signature in zip(functions, signatures):
        metric = CompiledMetric(
            pointer=cppyy.ll.addressof(function),
            kind=MetricKind.IP,
            signature=signature,
        )
        index = Index(ndim=ndim, metric=metric)

        labels = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)

        index.add(labels, vectors)
        matches, distances, count = index.search(vectors, 10)
        assert matches.shape[0] == distances.shape[0]
        assert count.shape[0] == batch_size

        assert recall_members(index, exact=True) == 1


@pytest.mark.parametrize("ndim", [8])
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_index_peachpy(ndim: int, batch_size: int):
    try:
        from peachpy import (
            Argument,
            ptr,
            float_,
            const_float_,
        )
        from peachpy.x86_64 import (
            abi,
            Function,
            uarch,
            isa,
            GeneralPurposeRegister64,
            LOAD,
            YMMRegister,
            VSUBPS,
            VADDPS,
            VHADDPS,
            VMOVUPS,
            VFMADD231PS,
            VPERM2F128,
            VXORPS,
            RETURN,
        )
    except ImportError:
        return

    a = Argument(ptr(const_float_), name="a")
    b = Argument(ptr(const_float_), name="b")

    with Function(
        "InnerProduct", (a, b), float_, target=uarch.default + isa.avx + isa.avx2
    ) as asm_function:
        # Request two 64-bit general-purpose registers for addresses
        reg_a, reg_b = GeneralPurposeRegister64(), GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_a, a)
        LOAD.ARGUMENT(reg_b, b)

        # Load the vectors
        ymm_a = YMMRegister()
        ymm_b = YMMRegister()
        VMOVUPS(ymm_a, [reg_a])
        VMOVUPS(ymm_b, [reg_b])

        # Prepare the accumulator
        ymm_c = YMMRegister()
        ymm_one = YMMRegister()
        VXORPS(ymm_c, ymm_c, ymm_c)
        VXORPS(ymm_one, ymm_one, ymm_one)

        # Accumulate A and B product into C
        VFMADD231PS(ymm_c, ymm_a, ymm_b)

        # Reduce the contents of a YMM register
        ymm_c_permuted = YMMRegister()
        VPERM2F128(ymm_c_permuted, ymm_c, ymm_c, 1)
        VADDPS(ymm_c, ymm_c, ymm_c_permuted)
        VHADDPS(ymm_c, ymm_c, ymm_c)
        VHADDPS(ymm_c, ymm_c, ymm_c)

        # Negate the values, to go from "similarity" to "distance"
        VSUBPS(ymm_c, ymm_one, ymm_c)

        # A common convention is to return floats in XMM registers
        RETURN(ymm_c.as_xmm)

    python_function = asm_function.finalize(abi.detect()).encode().load()
    metric = CompiledMetric(
        pointer=python_function.loader.code_address,
        kind=MetricKind.IP,
        signature=MetricSignature.ArrayArray,
    )
    index = Index(ndim=ndim, metric=metric)

    labels = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(labels, vectors)
    matches, distances, count = index.search(vectors, 10)
    assert matches.shape[0] == distances.shape[0]
    assert count.shape[0] == batch_size

    assert recall_members(index, exact=True) == 1


@pytest.mark.parametrize("bits", dimensions)
@pytest.mark.parametrize("metric", hash_metrics)
@pytest.mark.parametrize("connectivity", connectivity_options)
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_bitwise_index(
    bits: int, metric: MetricKind, connectivity: int, batch_size: int
):
    index = Index(ndim=bits, metric=metric, connectivity=connectivity)

    labels = np.arange(batch_size)
    byte_vectors = np.random.randint(2, size=(batch_size, bits))
    bit_vectors = np.packbits(byte_vectors, axis=1)

    index.add(labels, bit_vectors)
    assert np.allclose(index.get_vectors(labels), byte_vectors, atol=0.1)
    assert np.all(index.get_vectors(labels, ScalarKind.B1) == bit_vectors)

    index.search(bit_vectors, 10)

    if bits > batch_size:
        assert recall_members(index, exact=True) > 0.9


@pytest.mark.parametrize("connectivity", connectivity_options)
@pytest.mark.skipif(os.name == "nt", reason="Spurious behaviour on windows")
def test_sets_index(connectivity: int):
    index = SparseIndex(connectivity=connectivity)
    index.add(10, np.array([10, 12, 15], dtype=np.uint32))
    index.add(11, np.array([11, 12, 15, 16], dtype=np.uint32))
    results = index.search(np.array([12, 15], dtype=np.uint32), 10)
    assert list(results) == [10, 11]
