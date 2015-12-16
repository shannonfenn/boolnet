try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Distutils import Extension, build_ext
import numpy as np
import cython_gsl

include_dirs = [np.get_include(),
                cython_gsl.get_cython_include_dir()]


extensions = [
    Extension("boolnet.bintools.packing",
              ["boolnet/bintools/packing.pyx"],
              include_dirs=include_dirs),
    Extension("boolnet.bintools.bitcount",
              ["boolnet/bintools/bitcount.pyx"],
              libraries=['gmp'],
              include_dirs=include_dirs),
    Extension("boolnet.bintools.functions",
              ["boolnet/bintools/functions.pyx"],
              include_dirs=include_dirs),
    Extension("boolnet.bintools.biterror",
              ["boolnet/bintools/biterror.pyx"],
              include_dirs=include_dirs),
    Extension("boolnet.bintools.biterror_chained",
              ["boolnet/bintools/biterror_chained.pyx"],
              include_dirs=include_dirs),
    Extension("boolnet.bintools.operator_iterator",
              ["boolnet/bintools/operator_iterator.pyx"],
              include_dirs=include_dirs),
    Extension("boolnet.bintools.example_generator",
              ["boolnet/bintools/example_generator.pyx"],
              include_dirs=include_dirs),

    Extension("boolnet.exptools.fastrand",
              ["boolnet/exptools/fastrand.pyx"],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=include_dirs),

    Extension("boolnet.network.algorithms",
              ["boolnet/network/algorithms.pyx"],
              language='c++',
              include_dirs=include_dirs),
    Extension("boolnet.network.boolnet",
              ["boolnet/network/boolnet.pyx"],
              language='c++',
              include_dirs=include_dirs),

    Extension("boolnet.learning.networkstate",
              ["boolnet/learning/networkstate.pyx"],
              language='c++',
              include_dirs=include_dirs),
    ]

setup(
    name='boolnet',
    include_dirs=[np.get_include(), cython_gsl.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions
    )
