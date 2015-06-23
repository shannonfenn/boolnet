from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import os

setup(
    name='BoolNet',
    include_dirs=[cython_gsl.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("BoolNet.FastRand",
                  ["BoolNet/FastRand.pyx"],
                  libraries=cython_gsl.get_libraries(),
                  library_dirs=[cython_gsl.get_library_dir(),
                                os.path.expanduser('~/static/lib/')],
                  include_dirs=[cython_gsl.get_cython_include_dir(),
                                os.path.expanduser('~/static/include/')]),
        Extension("BoolNet.PopCount",
                  ["BoolNet/PopCount.pyx"],
                  libraries=['gmp'],
                  library_dirs=[os.path.expanduser('~/static/lib/')],
                  include_dirs=[os.path.expanduser('~/static/include/')]),
        Extension("BoolNet.Packing",
                  ["BoolNet/Packing.pyx"])]
    )
