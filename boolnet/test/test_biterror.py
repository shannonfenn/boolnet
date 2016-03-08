from boolnet.bintools.functions import function_name
from numpy.testing import assert_array_equal as assert_array_equal
from boolnet.bintools.biterror import STANDARD_EVALUATORS


def test_scalar_function(error_matrix_harness, scalar_function):
    Ne = error_matrix_harness['Ne']
    E = error_matrix_harness['packed error matrix']
    No, _ = E.shape
    eval_class, msb = STANDARD_EVALUATORS[scalar_function]
    error_evaluator = eval_class(Ne, No, msb)

    actual = error_evaluator.evaluate(E)
    expected = error_matrix_harness[function_name(scalar_function)]

    assert actual == expected


def test_per_output_function(error_matrix_harness, per_output_function):
    Ne = error_matrix_harness['Ne']
    E = error_matrix_harness['packed error matrix']
    No, _ = E.shape
    eval_class, msb = STANDARD_EVALUATORS[per_output_function]
    error_evaluator = eval_class(Ne, No, msb)

    actual = error_evaluator.evaluate(E)
    expected = error_matrix_harness[function_name(per_output_function)]

    assert_array_equal(actual, expected)


# @pytest.mark.python
# def test_error(error_matrix_harness, function):
#     from BoolNet.BitError import function_value
#     actual = function_value(error_matrix_harness['error matrix'], function)
#     expected = error_matrix_harness[function_name(function)]
#     assert np.array_equal(actual, expected)


# @pytest.mark.gpu
# def test_error_gpu(error_matrix_harness, function):
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     from BoolNet.BitErrorGPU import function_value_gpu, IMPLEMENTED_METRICS

#     err_mat = error_matrix_harness['error matrix']
#     d_error_mat = cuda.mem_alloc(err_mat.nbytes)
#     d_scratch_mat = cuda.mem_alloc(err_mat.shape[0] * 4)
#     Ne, No = err_mat.shape
#     Ne, No = np.int32(Ne), np.int32(No)
#     cuda.memcpy_htod(d_error_mat, err_mat)

#     expected = error_matrix_harness[function_name(function)]

#     if function in IMPLEMENTED_METRICS:
#         actual = function_value_gpu(d_error_mat, d_scratch_mat, Ne, No, function)
#         assert np.array_equal(actual, expected)
#     else:
#         with pytest.raises(NotImplementedError):
#             function_value_gpu(d_error_mat, d_scratch_mat, Ne, No, function)
#     d_scratch_mat.free()
#     d_error_mat.free()
