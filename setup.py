import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension


compile_args = ['-std=c++11', '-O3']

if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.13')

ext_modules = [
    Pybind11Extension(
        'usearch',
        ['src/python.cpp'],
        extra_compile_args=compile_args,
    ),
]


setup(

    name='usearch',
    version='0.1.0',
    packages=find_packages(),
    license='Apache-2.0',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License 2.0 (Apache-2.0)',

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

    include_dirs=['include', 'fp16/include', 'simsimd/include'],
    ext_modules=ext_modules,
)
