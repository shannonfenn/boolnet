from distutils.core import setup
from Cython.Build import cythonize
import os

os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

setup(name = 'pyBooleanNetworkNAND',
      ext_modules = cythonize('pyBooleanNetworkNAND.pyx'),
      )