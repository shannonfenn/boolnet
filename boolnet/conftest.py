import glob
import yaml
import numpy as np
import os.path
from copy import copy
from pytest import fixture
from bitpacking.packing import unpackmat
import boolnet.bintools.functions as fn


ERROR_MATRIX_FILES = glob.glob('boolnet/test/error matrices/*.yaml')
ERROR_MATRIX_CACHE = dict()


@fixture
def test_location():
    return 'boolnet/test/'


# #################### Fixtures ############################ #
@fixture(params=fn.all_function_names())
def any_function(request):
    return request.param


@fixture(params=fn.scalar_function_names())
def scalar_function(request):
    return request.param


@fixture(params=fn.per_output_function_names())
def per_output_function(request):
    return request.param


@fixture(params=ERROR_MATRIX_FILES)
def error_matrix_harness(request):
    fname = request.param
    if fname not in ERROR_MATRIX_CACHE:
        with open(request.param) as f:
            test = yaml.safe_load(f)
        folder = os.path.dirname(request.param)
        Ep = np.load(os.path.join(folder, test['name'] + '_E.npy'))
        Tp = np.load(os.path.join(folder, test['name'] + '_T.npy'))
        ERROR_MATRIX_CACHE[fname] = (test, Ep, Tp)
    # make copy of cached instance
    test, Ep, Tp = ERROR_MATRIX_CACHE[fname]
    test = copy(test)
    test['packed error matrix'] = np.array(Ep, copy=True)
    test['packed target matrix'] = np.array(Tp, copy=True)
    test['unpacked error matrix'] = unpackmat(Ep, test['Ne'])
    test['unpacked target matrix'] = unpackmat(Tp, test['Ne'])
    # No = Ep.shape[0]
    # test['mask'] = np.array(test.get('mask', [1]*No), dtype=np.uint8)
    return test
