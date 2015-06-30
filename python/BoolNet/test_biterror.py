from BoolNet.BitError import metric_name
import numpy as np
import pytest


@pytest.mark.python
def test_error(error_matrix_harness, metric):
    from BoolNet.BitError import metric_value
    actual = metric_value(error_matrix_harness['error matrix'], metric)
    expected = error_matrix_harness[metric_name(metric)]
    assert np.array_equal(actual, expected)


@pytest.mark.cython
def test_error_cython(error_matrix_harness, metric):
    import pyximport
    pyximport.install()
    from BoolNet.BitErrorCython import metric_value
    from BoolNet.Packing import generate_end_mask
    Ne = error_matrix_harness['Ne']
    E = error_matrix_harness['packed error matrix']
    E_out = np.zeros_like(E)
    end_mask = generate_end_mask(Ne)

    actual = metric_value(E, E_out, Ne, end_mask, metric)

    expected = error_matrix_harness[metric_name(metric)]

    assert np.array_equal(actual, expected)


@pytest.mark.gpu
def test_error_gpu(error_matrix_harness, metric):
    import pycuda.driver as cuda
    import pycuda.autoinit
    from BoolNet.BitErrorGPU import metric_value_gpu

    err_mat = error_matrix_harness['error matrix']
    d_error_matrix = cuda.mem_alloc(err_mat.nbytes)
    d_intermediate_matrix = cuda.mem_alloc(err_mat.shape[0] * 4)

    cuda.memcpy_htod(d_error_matrix, err_mat)

    actual = metric_value_gpu(d_error_matrix,
                              d_intermediate_matrix,
                              np.int32(err_mat.shape[0]),
                              np.int32(err_mat.shape[1]),
                              metric)
    expected = error_matrix_harness[metric_name(metric)]

    d_intermediate_matrix.free()
    d_error_matrix.free()

    assert np.array_equal(actual, expected)
