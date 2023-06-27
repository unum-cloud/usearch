import os
import sys
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension


compile_args = []
link_args = []
macros_args = [
    ("USEARCH_USE_NATIVE_F16", "0"),
    ("USEARCH_USE_SIMSIMD", "0"),
]

if sys.platform == "linux":
    compile_args.append("-std=c++11")
    compile_args.append("-O3")
    compile_args.append("-Wno-unknown-pragmas")

    macros_args.append(("USEARCH_USE_OPENMP", "1"))
    compile_args.append("-fopenmp")
    link_args.append("-lgomp")

if sys.platform == "darwin":
    # MacOS 10.15 or higher is needed for `aligned_alloc` support.
    # https://github.com/unum-cloud/usearch/actions/runs/4975434891/jobs/8902603392
    compile_args.append("-mmacosx-version-min=10.15")
    compile_args.append("-std=c++11")
    compile_args.append("-O3")
    compile_args.append("-Wno-unknown-pragmas")

    macros_args.append(("USEARCH_USE_OPENMP", "1"))
    compile_args.append("-Xpreprocessor -fopenmp")
    link_args.append("-Xpreprocessor -lomp")

if sys.platform == "win32":
    compile_args.append("/std:c++14")
    compile_args.append("/O2")

ext_modules = [
    Pybind11Extension(
        "usearch.compiled",
        ["python/lib.cpp"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
    ),
]

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "usearch"


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()


setup(
    name=__lib_name__,
    version=__version__,
    packages=["usearch"],
    package_dir={"usearch": "python/usearch"},
    description="Smaller & Faster Single-File Vector Search Engine from Unum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Java",
        "Programming Language :: JavaScript",
        "Programming Language :: Objective C",
        "Programming Language :: Rust",
        "Programming Language :: Other",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Topic :: System :: Clustering",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_dirs=["include", "fp16/include", "robin-map/include", "simsimd/include"],
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        'ucall; python_version >= "3.9"',
    ],
)
