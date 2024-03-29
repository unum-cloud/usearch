import pytest
import numpy as np

from usearch.eval import random_vectors

from usearch.index import (
    Index,
    MetricKind,
    MetricSignature,
    CompiledMetric,
)

dimensions = [3, 97, 256]
batch_sizes = [1, 77]


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_index_numba(ndim: int, batch_size: int):
    try:
        from numba import cfunc, types, carray
    except ImportError:
        pytest.skip("Numba is not installed.")
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

    functions = [
        python_inner_product_two_args,
        python_inner_product_three_args,
    ]
    signatures = [
        MetricSignature.ArrayArray,
        MetricSignature.ArrayArraySize,
    ]
    for function, signature in zip(functions, signatures):
        metric = CompiledMetric(
            pointer=function.address,
            kind=MetricKind.IP,
            signature=signature,
        )
        index = Index(ndim=ndim, metric=metric, dtype=np.float32)

        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)

        index.add(keys, vectors)
        matches = index.search(vectors, 10, exact=True)
        assert len(matches) == batch_size

        matches_keys = [match[0].key for match in matches] if batch_size > 1 else [matches[0].key]
        assert all(matches_keys[i] == keys[i] for i in range(batch_size)), f"Received {matches_keys}"


# Just one size for Cppyy to avoid redefining kernels in the global namespace
@pytest.mark.parametrize("ndim", dimensions[-1:])
@pytest.mark.parametrize("batch_size", batch_sizes[-1:])
def test_index_cppyy(ndim: int, batch_size: int):
    try:
        import cppyy
        import cppyy.ll
    except ImportError:
        pytest.skip("cppyy is not installed.")
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
    """.replace(
            "ndim", str(ndim)
        )
    )

    functions = [
        cppyy.gbl.inner_product_two_args,
        cppyy.gbl.inner_product_three_args,
    ]
    signatures = [
        MetricSignature.ArrayArray,
        MetricSignature.ArrayArraySize,
    ]
    for function, signature in zip(functions, signatures):
        metric = CompiledMetric(
            pointer=cppyy.ll.addressof(function),
            kind=MetricKind.IP,
            signature=signature,
        )
        index = Index(ndim=ndim, metric=metric, dtype=np.float32)

        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim, dtype=np.float32)

        index.add(keys, vectors)
        matches = index.search(vectors, 10, exact=True)
        assert len(matches) == batch_size

        matches_keys = [match[0].key for match in matches] if batch_size > 1 else [matches[0].key]
        assert all(matches_keys[i] == keys[i] for i in range(batch_size)), f"Received {matches_keys}"


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
        pytest.skip("PeachPy is not installed.")
        return

    a = Argument(ptr(const_float_), name="a")
    b = Argument(ptr(const_float_), name="b")

    with Function("InnerProduct", (a, b), float_, target=uarch.default + isa.avx + isa.avx2) as asm_function:
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
    index = Index(ndim=ndim, metric=metric, dtype=np.float32)

    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors)
    matches = index.search(vectors, 10, exact=True)
    assert len(matches) == batch_size

    matches_keys = [match[0].key for match in matches] if batch_size > 1 else [matches[0].key]
    assert all(matches_keys[i] == keys[i] for i in range(batch_size)), f"Received {matches_keys}"
