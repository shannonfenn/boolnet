from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy as np
import cython_gsl
import os

extensions = [
    Extension("boolnet.bintools.packing", ["boolnet/bintools/packing.pyx"]),
    Extension("boolnet.bintools.bitcount", ["boolnet/bintools/bitcount.pyx"],
              libraries=['gmp'],
              library_dirs=[os.path.expanduser('~/static/lib/')],
              include_dirs=[os.path.expanduser('~/static/include/')]),
    Extension("boolnet.bintools.functions", ["boolnet/bintools/functions.pyx"]),
    Extension("boolnet.bintools.biterror", ["boolnet/bintools/biterror.pyx"]),
    Extension("boolnet.bintools.biterror_chained", ["boolnet/bintools/biterror_chained.pyx"]),
    Extension("boolnet.bintools.operator_iterator", ["boolnet/bintools/operator_iterator.pyx"]),
    Extension("boolnet.bintools.example_generator", ["boolnet/bintools/example_generator.pyx"]),

    Extension("boolnet.exptools.fastrand", ["boolnet/exptools/fastrand.pyx"],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=[cython_gsl.get_cython_include_dir()],),

    Extension("boolnet.network.algorithms", ["boolnet/network/algorithms.pyx"],
              language='c++'),
    Extension("boolnet.network.boolnetwork", ["boolnet/network/boolnetwork.pyx"],
              language='c++'),

    Extension("boolnet.learning.networkstate", ["boolnet/learning/networkstate.pyx"],
              language='c++'),
    ]

setup(
    name='boolnet',
    include_dirs=[np.get_include(), cython_gsl.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions
    )
