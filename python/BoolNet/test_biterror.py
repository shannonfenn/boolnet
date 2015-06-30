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
    from BoolNet.BitErrorGPU import metric_value_gpu, IMPLEMENTED_METRICS

    err_mat = error_matrix_harness['error matrix']
    d_error_mat = cuda.mem_alloc(err_mat.nbytes)
    d_scratch_mat = cuda.mem_alloc(err_mat.shape[0] * 4)
    Ne, No = err_mat.shape
    Ne, No = np.int32(Ne), np.int32(No)
    cuda.memcpy_htod(d_error_mat, err_mat)

    expected = error_matrix_harness[metric_name(metric)]

    if metric in IMPLEMENTED_METRICS:
        actual = metric_value_gpu(d_error_mat, d_scratch_mat, Ne, No, metric)
        assert np.array_equal(actual, expected)
    else:
        with pytest.raises(NotImplementedError):
            metric_value_gpu(d_error_mat, d_scratch_mat, Ne, No, metric)
    d_scratch_mat.free()
    d_error_mat.free()
