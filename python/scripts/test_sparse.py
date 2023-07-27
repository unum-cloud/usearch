import os

import pytest
import numpy as np

from usearch.index import (
    SparseIndex,
)
from usearch.index import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_EXPANSION_ADD,
    DEFAULT_EXPANSION_SEARCH,
)


dimensions = [3, 97, 256]
batch_sizes = [1, 77]
connectivity_options = [3, 13, 50, DEFAULT_CONNECTIVITY]


@pytest.mark.parametrize("connectivity", connectivity_options)
@pytest.mark.skipif(os.name == "nt", reason="Spurious behaviour on windows")
def test_sets_index(connectivity: int):
    index = SparseIndex(connectivity=connectivity)
    index.add(10, np.array([10, 12, 15], dtype=np.uint32))
    index.add(11, np.array([11, 12, 15, 16], dtype=np.uint32))
    results = index.search(np.array([12, 15], dtype=np.uint32), 10)
    assert list(results) == [10, 11]


if __name__ == "__main__":
    pytest.main(args=["python/scripts/test.py", "-s", "-x", "-v"])
