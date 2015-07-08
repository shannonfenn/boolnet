import glob
import yaml
import numpy as np
import os.path
from copy import copy
from pytest import fixture
from boolnet.bintools.packing import unpack_bool_matrix
from boolnet.bintools.metric_names import all_metrics


ERROR_MATRIX_FILES = glob.glob('boolnet/test/error matrices/*.yaml')
ERROR_MATRIX_CACHE = dict()


@fixture
def test_location():
    return 'boolnet/test/'


# #################### Fixtures ############################ #
@fixture(params=list(all_metrics()))
def metric(request):
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
    return test
