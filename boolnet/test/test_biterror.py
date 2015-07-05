from boolnet.bintools.metric_names import metric_name
import numpy as np
from numpy.testing import assert_array_equal as assert_array_equal
from numpy.testing import assert_array_almost_equal as assert_array_almost_equal
import pyximport
pyximport.install()
from boolnet.bintools.biterror_chained import CHAINED_EVALUATORS
from boolnet.bintools.biterror import STANDARD_EVALUATORS
from boolnet.bintools.packing import packed_type


# @pytest.mark.python
# def test_error(error_matrix_harness, metric):
#     from BoolNet.BitError import metric_value
#     actual = metric_value(error_matrix_harness['error matrix'], metric)
#     expected = error_matrix_harness[metric_name(metric)]
#     assert np.array_equal(actual, expected)

def eval_chained(window_width, E, error_evaluator):
    No, array_width = E.shape
    window = np.array(np.zeros(shape=(No, window_width)), dtype=packed_type)

    steps = array_width // window_width
    if array_width % window_width != 0:
        steps += 1

    for i in range(steps-1):
        window[:, :] = E[:, i*window_width: (i+1)*window_width]
        error_evaluator.partial_evaluation(window)

    if array_width % window_width != 0:
        print(array_width, window_width, array_width % window_width)
        print(steps, (steps-1)*window_width)
        window[:, :] = 0
        window[:, :array_width % window_width] = np.array(E[:, (steps-1)*window_width:])
    else:
        window[:, :] = E[:, -window_width:]

    return error_evaluator.final_evaluation(window)


def test_biterror_chained(packed_error_matrix_harness, metric):
    Ne = packed_error_matrix_harness['Ne']
    E = packed_error_matrix_harness['packed error matrix']
    No, cols = E.shape

    eval_class, msb = CHAINED_EVALUATORS[metric]
    widths = np.random.randint(1, cols, size=5)
    for window_width in widths:
        error_evaluator = eval_class(Ne, No, window_width, msb)
        actual = eval_chained(window_width, E, error_evaluator)
        expected = packed_error_matrix_harness[metric_name(metric)]
        assert_array_almost_equal(actual, expected)


def test_biterror_packed(packed_error_matrix_harness, metric):
    Ne = packed_error_matrix_harness['Ne']
    E = packed_error_matrix_harness['packed error matrix']
    No, _ = E.shape
    eval_class, msb = STANDARD_EVALUATORS[metric]
    error_evaluator = eval_class(Ne, No, msb)

    actual = error_evaluator.evaluate(E)
    expected = packed_error_matrix_harness[metric_name(metric)]

    assert_array_equal(actual, expected)


def test_biterror(error_matrix_harness, metric):
    Ne = error_matrix_harness['Ne']
    E = error_matrix_harness['packed error matrix']
    No, _ = E.shape
    eval_class, msb = STANDARD_EVALUATORS[metric]
    error_evaluator = eval_class(Ne, No, msb)

    actual = error_evaluator.evaluate(E)
    expected = error_matrix_harness[metric_name(metric)]

    assert_array_equal(actual, expected)


# @pytest.mark.gpu
# def test_error_gpu(error_matrix_harness, metric):
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     from BoolNet.BitErrorGPU import metric_value_gpu, IMPLEMENTED_METRICS

#     err_mat = error_matrix_harness['error matrix']
#     d_error_mat = cuda.mem_alloc(err_mat.nbytes)
#     d_scratch_mat = cuda.mem_alloc(err_mat.shape[0] * 4)
#     Ne, No = err_mat.shape
#     Ne, No = np.int32(Ne), np.int32(No)
#     cuda.memcpy_htod(d_error_mat, err_mat)

#     expected = error_matrix_harness[metric_name(metric)]

#     if metric in IMPLEMENTED_METRICS:
#         actual = metric_value_gpu(d_error_mat, d_scratch_mat, Ne, No, metric)
#         assert np.array_equal(actual, expected)
#     else:
#         with pytest.raises(NotImplementedError):
#             metric_value_gpu(d_error_mat, d_scratch_mat, Ne, No, metric)
#     d_scratch_mat.free()
#     d_error_mat.free()
