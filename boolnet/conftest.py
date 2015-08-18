import glob
import yaml
import numpy as np
import os.path
from copy import copy
from pytest import fixture
from boolnet.bintools.packing import unpack_bool_matrix
import pyximport
pyximport.install()
from boolnet.bintools.functions import all_functions


ERROR_MATRIX_FILES = glob.glob('boolnet/test/error matrices/*.yaml')
ERROR_MATRIX_CACHE = dict()


@fixture
def test_location():
    return 'boolnet/test/'


# #################### Fixtures ############################ #
@fixture(params=all_functions())
def function(request):
    return request.param


@fixture(params=ERROR_MATRIX_FILES)
def error_matrix_harness(request):
    fname = request.param
    if fname not in ERROR_MATRIX_CACHE:
        with open(request.param) as f:
            test = yaml.safe_load(f)
        folder = os.path.dirname(request.param)
        Ep = np.load(os.path.join(folder, test['name'] + '.npy'))
        ERROR_MATRIX_CACHE[fname] = (test, Ep)
    # make copy of cached instance
    test, Ep = ERROR_MATRIX_CACHE[fname]
    test = copy(test)
    Ep = np.array(Ep, copy=True)
    E = unpack_bool_matrix(Ep, test['Ne'])
    test['packed error matrix'] = Ep
    test['unpacked error matrix'] = E
    # No = Ep.shape[0]
    # test['mask'] = np.array(test.get('mask', [1]*No), dtype=np.uint8)
    return test
