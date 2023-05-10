import os
import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension


compile_args = ['-std=c++11', '-O3']

if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.13')

ext_modules = [
    Pybind11Extension(
        'usearch.index',
        ['python/lib.cpp'],
        extra_compile_args=compile_args
    ),
]

__version__ = open('VERSION', 'r').read().strip()
__lib_name__ = 'usearch'


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()


setup(

    name=__lib_name__,
    version=__version__,
    packages=['usearch'],
    package_dir={'usearch': 'python/usearch'},

    description='Smaller & Faster Single-File Vector Search Engine from Unum',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: C++',

        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',

        'Topic :: System :: Clustering',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    include_dirs=['include', 'fp16/include', 'simsimd/include', 'src'],
    ext_modules=ext_modules,
)
