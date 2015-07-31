import numpy as np
from pytest import fixture
from numpy.testing import assert_array_almost_equal as assert_array_almost_equal
from boolnet.bintools.metrics import metric_name
from boolnet.bintools.biterror_chained import CHAINED_EVALUATORS
from boolnet.bintools.packing import packed_type


@fixture
def single_column(error_matrix_harness, metric):
    error_matrix_harness['metric'] = metric
    window_width = 1
    error_matrix_harness['window_width'] = window_width
    return construct_test_instance(error_matrix_harness)


@fixture
def full_width(error_matrix_harness, metric):
    Ep = error_matrix_harness['packed error matrix']
    error_matrix_harness['metric'] = metric
    window_width = Ep.shape[1]
    error_matrix_harness['window_width'] = window_width
    return construct_test_instance(error_matrix_harness)


@fixture
def random_width(error_matrix_harness, metric):
    Ep = error_matrix_harness['packed error matrix']
    error_matrix_harness['metric'] = metric
    window_width = max(int(np.random.random(1) * Ep.shape[1]), 1)
    error_matrix_harness['window_width'] = window_width
    return construct_test_instance(error_matrix_harness)


def construct_test_instance(harness):
    Ep = harness['packed error matrix']
    metric = harness['metric']
    window_width = harness['window_width']
    mask = harness['mask']

    Ne = harness['Ne']
    No, cols = Ep.shape

    eval_class, msb = CHAINED_EVALUATORS[metric]
    error_evaluator = eval_class(Ne, No, window_width, msb, mask)

    expected = harness[metric_name(metric)]

    print(metric_name(metric))

    return (window_width, Ep, error_evaluator, expected)


def eval_chained(window_width, E, error_evaluator):
    ''' This helper evaluates the error matrix using the given
        chained evaluator.'''
    No, array_width = E.shape
    window = np.array(np.zeros(shape=(No, window_width)), dtype=packed_type)

    steps = array_width // window_width
    if array_width % window_width != 0:
        steps += 1

    for i in range(steps-1):
        window[:, :] = E[:, i*window_width: (i+1)*window_width]
        error_evaluator.partial_evaluation(window)

    if array_width % window_width != 0:
        window[:, :] = 0
        window[:, :array_width % window_width] = np.array(E[:, (steps-1)*window_width:])
    else:
        window[:, :] = E[:, -window_width:]

    return error_evaluator.final_evaluation(window)


def test_single_column_eval(single_column):
    ''' Test the chained evaluator gives correct metric values
        for varying window widths.'''
    window_width, E, error_evaluator, expected = single_column

    actual = eval_chained(window_width, E, error_evaluator)
    assert_array_almost_equal(expected, actual)


def test_full_width_eval(full_width):
    ''' Test the chained evaluator gives correct metric values
        for varying window widths.'''
    window_width, E, error_evaluator, expected = full_width

    actual = eval_chained(window_width, E, error_evaluator)
    assert_array_almost_equal(expected, actual)


def test_eval(random_width):
    ''' Test the chained evaluator gives correct metric values
        for varying window widths.'''
    window_width, E, error_evaluator, expected = random_width

    actual = eval_chained(window_width, E, error_evaluator)
    assert_array_almost_equal(expected, actual)


def test_reset(random_width):
    ''' Test the chained evaluator gives correct metric values
        after being evaluated and reset.'''
    window_width, E, error_evaluator, expected = random_width

    # evaluate once and then reset
    eval_chained(window_width, E, error_evaluator)
    error_evaluator.reset()

    actual = eval_chained(window_width, E, error_evaluator)
    assert_array_almost_equal(actual, expected)
