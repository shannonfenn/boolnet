import numpy as np
from numpy.testing import assert_array_equal
import pytest
import boolnet.network.networkstate as ns
import boolnet.network.algorithms as alg
import boolnet.utils as utils
import bitpacking.packing as pk


# ############## Fixtures ##################### #
@pytest.fixture
def random_state():
    # generate random feedforward network and problem matrix
    Ni = np.random.randint(1, 10)
    No = np.random.randint(1, 10)
    Ng = np.random.randint(No, 20)
    Ne = np.random.randint(1, 100)

    gates = [(np.random.randint(g+Ni),
              np.random.randint(g+Ni),
              np.random.randint(16))
             for g in range(Ng)]

    P = np.random.randint(0, 2, size=(Ne, Ni+No), dtype=np.uint8)
    P = utils.PackedMatrix(pk.packmat(P), Ne, Ni)
    return gates, P


# #################### Functional Testing #################### #

@pytest.mark.parametrize('repeats', range(10))
def test_filter_connected(repeats, random_state):
    gates, problem_matrix = random_state
    state = ns.BNState(gates, problem_matrix)
    expected_O = np.array(state.output_matrix)
    expected_E = np.array(state.error_matrix)
    expected_Ng = np.flatnonzero(np.asarray(state.connected_gates())).size

    new_gates = alg.filter_connected(state.gates, state.Ni, state.No)

    print(problem_matrix.Ni, problem_matrix.No)
    print(state.gates.shape)
    print(new_gates.shape)
    print(new_gates)

    new_state = ns.BNState(new_gates, problem_matrix)

    assert new_state.Ng == expected_Ng
    assert_array_equal(new_state.output_matrix, expected_O)
    assert_array_equal(new_state.error_matrix, expected_E)
