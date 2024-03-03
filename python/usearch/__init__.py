import importlib
import importlib.util

from usearch.compiled import (
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
)

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

# The same binary file (.so, .dll, or .dylib) that contains the pre-compiled
# USearch code also contains the SQLite3 binding
sqlite = importlib.util.find_spec("usearch.compiled").origin
