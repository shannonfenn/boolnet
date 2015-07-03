from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl


from Cython.Compiler.Options import directive_defaults
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True


setup(
    name='BoolNet',
    include_dirs=[cython_gsl.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("BoolNet.FastRand",
                  ["BoolNet/FastRand.pyx"],
                  libraries=cython_gsl.get_libraries(),
                  library_dirs=[cython_gsl.get_library_dir()],
                  include_dirs=[cython_gsl.get_cython_include_dir()],
                  define_macros=[('CYTHON_TRACE', '1')]),
        Extension("BoolNet.PopCount",
                  ["BoolNet/PopCount.pyx"],
                  libraries=['gmp'],
                  define_macros=[('CYTHON_TRACE', '1')]),
        Extension("BoolNet.Packing",
                  ["BoolNet/Packing.pyx"],
                  define_macros=[('CYTHON_TRACE', '1')])
        ],
    )
