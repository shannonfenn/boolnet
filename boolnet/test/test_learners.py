import pytest
import numpy as np
import minfs.feature_selection as mfs
import boolnet.learners.classifierchain as cc
from boolnet.network.networkstate import BNState
from boolnet.utils import PackedMatrix


def random_network(Ng, Ni, No, node_funcs):
    # generate random feedforward network
    gates = np.empty(shape=(Ng, 3), dtype=np.int32)
    for g in range(Ng):
        # don't allow connecting outputs together
        gates[g, 0] = np.random.randint(min(g, Ng - No) + Ni)
        gates[g, 1] = np.random.randint(min(g, Ng - No) + Ni)
    gates[:, 2] = np.random.choice(node_funcs, size=Ng)
    return gates


@pytest.mark.parametrize('execution_number', range(25))
def test_classifierchain_join_networks(execution_number):
    Ni = np.random.randint(1, 20)
    No = np.random.randint(1, 20)
    Ng = np.random.randint(1, 20)
    Ne = np.random.randint(8, 256)
    Ncol = int(np.ceil(Ne / 64))

    print(Ni, No, Ng)

    X = np.random.randint(2**64, size=(Ni, Ncol), dtype=np.uint64)
    Y = np.random.randint(2**64, size=(No, Ncol), dtype=np.uint64)
    D = PackedMatrix(np.vstack((X, Y)), Ni=Ni, Ne=Ne)

    nets = [random_network(Ng, Ni + i, 1, list(range(16)))
            for i in range(No)]

    print(nets)

    curriculum = np.random.permutation(No)
    inv_curriculum = mfs.inverse_permutation(curriculum)

    outputs = []
    states = []
    # for i, gates in enumerate(nets):
    for i, gates in zip(curriculum, nets):
        Dsub = np.vstack([X] + outputs + [Y[[i], :]])
        Dsub = PackedMatrix(Dsub, Ni=Ni+len(outputs), Ne=Ne)
        net = BNState(gates, Dsub)
        outputs.append(np.array(net.output_matrix))
        states.append(net)

    # expected = np.vstack(expected)
    expected = np.vstack([outputs[i] for i in inv_curriculum])

    # combined = cc.join_networks(states, list(range(No)))
    combined = cc.join_networks(states, inv_curriculum)

    print(combined)

    combined = BNState(combined.gates, D)
    actual = np.array(combined.output_matrix)

    np.testing.assert_array_equal(expected, actual)
