from bitpacking import packing as pk
from boolnet.bintools import functions as fn
from boolnet.bintools import biterror as be
import numpy as np
from sklearn.metrics import matthews_corrcoef
from pytest import mark


def constant_columns(X):
    return np.logical_or((X == 0).all(axis=0), (X == 1).all(axis=0))


def test_function_value_harness(error_matrix_harness):
    Ne = error_matrix_harness['Ne']
    E = error_matrix_harness['packed error matrix']
    T = error_matrix_harness['packed target matrix']
    No, _ = E.shape

    for test in error_matrix_harness['tests']:
        order = test['order']
        func_id = fn.function_from_name(test['function'])
        expected = test['value']

        if order == 'lsb':
            order = np.arange(No, dtype=np.uintp)
        elif order == 'msb':
            order = np.arange(No, dtype=np.uintp)[::-1]

        E_ = E[order, :]
        T_ = T[order, :]

        eval_class = be.EVALUATORS[func_id]

        error_evaluator = eval_class(Ne, No)
        actual = error_evaluator.evaluate(E_, T_)
        np.testing.assert_array_almost_equal(actual, expected)


# will do pairwise combinations so this is quadratic
@mark.parametrize('Ne', np.random.randint(2, 100, size=5))
@mark.parametrize('No', np.random.randint(1, 100, size=5))
def test_mcc_random(Ne, No):
    O = np.random.randint(2, size=(Ne, No))
    T = np.random.randint(2, size=(Ne, No))

    const_cols_O = constant_columns(O)
    const_cols_T = constant_columns(T)
    for col in np.flatnonzero(const_cols_O):
        # toggle one random position in each constant column
        mask = np.random.permutation(np.eye(Ne)[0])
        O[:, col] = np.logical_xor(O[:, col], mask)
    for col in np.flatnonzero(const_cols_T):
        # toggle one random position in each constant column
        mask = np.random.permutation(np.eye(Ne)[0])
        T[:, col] = np.logical_xor(T[:, col], mask)

    E = np.logical_xor(O, T)

    mccs = [matthews_corrcoef(y_pred=o, y_true=t) for o, t in zip(O.T, T.T)]

    expected_values = {
        fn.PER_OUTPUT_MCC: mccs,
        fn.MACRO_MCC: np.mean(mccs)
    }

    Ep = pk.packmat(E.astype(dtype=np.uint8))
    Tp = pk.packmat(T.astype(dtype=np.uint8))

    for func_id, expected in expected_values.items():
        eval_class = be.EVALUATORS[func_id]
        error_evaluator = eval_class(Ne, No)
        actual = error_evaluator.evaluate(Ep, Tp)
        np.testing.assert_array_almost_equal(actual, expected)


def test_e3_general():
    E = np.array([[1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0]], dtype=np.uint8)
    Ne, No = E.shape
    Ep = pk.packmat(E)
    Tp = np.zeros_like(Ep)

    eval_class = be.EVALUATORS[fn.function_from_name('e3_general')]

    dependencies = [[], [], [0, 1], [0, 1, 2], [0, 1, 2]]

    error_evaluator = eval_class(Ne, No, dependencies)
    actual = error_evaluator.evaluate(Ep, Tp)
    np.testing.assert_array_almost_equal(actual, 27/45)


def test_e6_general():
    E = np.array([[0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 1]], dtype=np.uint8)
    Ne, No = E.shape
    Ep = pk.packmat(E)
    Tp = np.zeros_like(Ep)

    eval_class = be.EVALUATORS[fn.function_from_name('e6_general')]

    dependencies = [[], [], [0, 1], [0, 1, 2], [0, 1, 2], [0]]

    error_evaluator = eval_class(Ne, No, dependencies)
    actual = error_evaluator.evaluate(Ep, Tp)
    np.testing.assert_array_almost_equal(actual, 36/54)
