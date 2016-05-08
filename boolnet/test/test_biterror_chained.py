import numpy as np
from pytest import fixture
from numpy.testing import assert_array_almost_equal
from boolnet.bintools.functions import function_from_name
from boolnet.bintools.biterror_chained import CHAINED_EVALUATORS
from boolnet.bintools.packing import packed_type


def eval_chained(window_width, Ep, error_evaluator):
    ''' This helper evaluates the error matrix using the given
        chained evaluator.'''
    No, array_width = Ep.shape
    # error window
    Ew = np.array(np.zeros((No, window_width)), dtype=packed_type)
    Tw = np.zeros_like(Ew)

    steps = array_width // window_width
    if array_width % window_width != 0:
        steps += 1

    for i in range(steps-1):
        Ew[:, :] = Ep[:, i*window_width: (i+1)*window_width]
        error_evaluator.partial_evaluation(Ew, Tw)

    if array_width % window_width != 0:
        Ew[:, :] = 0
        Ew[:, :array_width % window_width] = np.array(
            Ep[:, (steps-1)*window_width:])
    else:
        Ew[:, :] = Ep[:, -window_width:]

    return error_evaluator.final_evaluation(Ew, Tw)


def evaluate(harness, window_width):
    Ep = harness['packed error matrix']
    Ne = harness['Ne']
    No, _ = Ep.shape
    for test in harness['tests']:
        print(test)
        func_name = test['function']
        order = test['order']
        expected = test['value']

        if order == 'l':
            order = np.arange(No, dtype=np.uintp)
        elif order == 'm':
            order = np.arange(No, dtype=np.uintp)[::-1]

        eval_class = CHAINED_EVALUATORS[function_from_name(func_name)]
        error_evaluator = eval_class(Ne, Ep.shape[0], window_width, order)

        actual = eval_chained(window_width, Ep, error_evaluator)
        assert_array_almost_equal(expected, actual)


def test_single_column_eval(error_matrix_harness):
    ''' Test the chained evaluator gives correct function values
        for a window width of 1.'''
    window_width = 1
    evaluate(error_matrix_harness, window_width)


def test_full_width_eval(error_matrix_harness):
    ''' Test the chained evaluator gives correct function values
        for full window width.'''
    Ep = error_matrix_harness['packed error matrix']
    window_width = Ep.shape[1]
    evaluate(error_matrix_harness, window_width)


def test_eval(error_matrix_harness):
    ''' Test the chained evaluator gives correct function values
        for varying window widths.'''
    Ep = error_matrix_harness['packed error matrix']
    window_width = max(int(np.random.random(1) * Ep.shape[1]), 1)
    evaluate(error_matrix_harness, window_width)


def test_reset(error_matrix_harness):
    ''' Test the chained evaluator gives correct function values
        after being evaluated and reset.'''
    # random width
    Ep = error_matrix_harness['packed error matrix']
    window_width = max(int(np.random.random(1) * Ep.shape[1]), 1)

    Ne = error_matrix_harness['Ne']
    No, _ = Ep.shape
    for test in error_matrix_harness['tests']:
        print(test)
        func_name = test['function']
        order = test['order']
        expected = test['value']

        if order == 'l':
            order = np.arange(No, dtype=np.uintp)
        elif order == 'm':
            order = np.arange(No, dtype=np.uintp)[::-1]

        eval_class = CHAINED_EVALUATORS[function_from_name(func_name)]
        error_evaluator = eval_class(Ne, Ep.shape[0], window_width, order)

        # evaluate once and then reset
        eval_chained(window_width, Ep, error_evaluator)
        error_evaluator.reset()
        actual = eval_chained(window_width, Ep, error_evaluator)
        # actual = eval_chained(1, Ep, error_evaluator)

        assert_array_almost_equal(expected, actual)
