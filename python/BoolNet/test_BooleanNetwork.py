from copy import copy, deepcopy
from itertools import chain
import numpy as np
from numpy.random import randint
from BoolNet.BooleanNetwork import BooleanNetwork
import networkx as nx
import pytest
xfail = pytest.mark.xfail


# ############# General helpers ################# #
def to_binary(value, num_bits):
    return [int(i) for i in '{:0{w}b}'.format(value, w=num_bits)][::-1]


def all_possible_inputs(num_bits):
    return np.array([to_binary(i, num_bits) for i in range(2**num_bits)],
                    dtype=np.uint8)


def gates_to_NX_digraph(gates, Ni, No):
    DG = nx.DiGraph()
    DG.add_nodes_from(range(Ni))
    edges = []
    for g in range(len(gates)):
        edges.append((gates[g][0], g+Ni))
        edges.append((gates[g][1], g+Ni))
    DG.add_edges_from(edges)
    return DG


def connected_ground_truth(net):
    Ni = net.Ni
    No = net.No
    Ng = net.Ng
    digraph = gates_to_NX_digraph(net.gates, Ni, No)
    connected = set(chain.from_iterable(nx.ancestors(digraph, Ni+Ng-o-1)
                    for o in range(No)))
    connected = set([g - Ni for g in connected if g >= Ni])
    connected = connected.union(range(Ng-No, Ng))
    return connected


# ############## Fixtures ##################### #

class TestExceptions:
    invalid_constructions = [
        ([], 0, 0),             # all invalid
        ([], 0, 1),             # gates and Ni invalid, No > Ng
        ([], 1, 0),             # gates and No invalid
        ([], 1, 1),             # gates invalid, No > Ng
        ([[0, 0]], 0, 0),       # Ni and No invalid
        ([[0, 0]], 1, 0),       # No invalid
        ([[0, 0]], 0, 1),       # Ni invalid
        ([[0, 0]], 1, 2),       # No > Ng
        ([0, 0], 1, 1),         # gates not 2D
        ([[]], 0, 1),           # gates not mx2
        ([[0]], 0, 1),          # gates not mx2
        ([[0, 0, 0]], 0, 1)]    # gates not mx2

    # for adder2 test Ni = 4
    invalid_masks = [
        ([], []),           # sourceable and changeable empty
        ([0], []),          # changeable empty
        ([], [0]),          # sourceable empty
        ([4], [0]),         # sourceable nodes == changeable gate
        ([7, 8, 9], [0]),   # sourceable nodes > changeable gate
        ([4, 5, 6], [0])]   # sourceable nodes >= changeable gate

    @pytest.fixture
    def network(self):
        return BooleanNetwork([(0, 1)], 2, 1)

    @pytest.fixture(params=invalid_constructions)
    def invalid_construction(self, request):
        return request.param

    @pytest.fixture(params=invalid_masks)
    def invalid_mask(self, request):
        return request.param

    ''' Exception Testing '''
    def test_construction_exceptions(self, invalid_construction):
        with pytest.raises(ValueError):
            BooleanNetwork(*invalid_construction)

    def test_set_mask_exceptions(self, adder2, invalid_mask):
        # the following have no valid sourceable nodes for any of the
        # changeable ones - remember 2 bit adders means sources 0-3 are
        # inputs and only source 4-> are gates
        # this means that ([1,2,3], [0]) is valid since gate 0, is source 4
        net = adder2['network']

        with pytest.raises(ValueError):
            net.set_mask(*invalid_mask)

    def test_initial_revert_move_exception(self, network):
        with pytest.raises(RuntimeError):
            network.revert_move()

    def test_1_move_2_revert_exception(self, network):
        move = network.random_move()
        network.move_to_neighbour(move)
        network.revert_move()
        with pytest.raises(RuntimeError):
            network.revert_move()

    def test_n_move_n_plus_2_revert_exception(self, network):
        # n moves, n + 2 reverts
        for i in range(10):
            network.move_to_neighbour(network.random_move())
        for i in range(10):
            network.revert_move()
        with pytest.raises(RuntimeError):
            network.revert_move()
        with pytest.raises(RuntimeError):
            network.revert_move()

    def test_revert_after_revert_all_exception(self, network):
        # revert after revert all
        for i in range(10):
            network.move_to_neighbour(network.random_move())
        network.revert_all_moves()
        with pytest.raises(RuntimeError):
            network.revert_move()


# #################### Functional Testing #################### #
def test_connected_gates():
    # generate random feedforward network
    Ni = randint(1, 10)
    No = randint(1, 10)
    Ng = randint(No, 20)
    random_gates = [(randint(0, g+Ni), randint(0, g+Ni)) for g in range(Ng)]
    # create the seed network
    net = BooleanNetwork(random_gates, Ni, No)

    actual = net.connected_gates(No)
    DG = gates_to_NX_digraph(random_gates, Ni, No)

    assert nx.is_directed_acyclic_graph(DG)

    expected = set(chain.from_iterable(nx.ancestors(DG, Ni+Ng-o-1)
                   for o in range(No)))
    expected = np.array([g - Ni for g in expected if g >= Ni])
    expected = np.unique(np.concatenate((expected, np.arange(Ng-No, Ng))))

    assert np.array_equal(actual, expected)


def test_revert_move_revert(any_test_network):
    net = any_test_network['network']
    for i in range(100):
        old_gates = copy(net.gates)
        net.move_to_neighbour(net.random_move())
        assert not np.array_equal(net.gates, old_gates)
        net.revert_move()
        assert np.array_equal(net.gates, old_gates)


def test_revert_move_revert_multi(any_test_network):
    net = any_test_network['network']
    backups = []
    # move network 30 times in a row and ensure it is different every time
    # keeping a list of the networks
    for i in range(30):
        backups.append(deepcopy(net))
        net.move_to_neighbour(net.random_move())
        assert not np.array_equal(net.gates, backups[-1].gates)
    # revert it incrementally and check it is equal to the backup each time
    for backup in reversed(backups):
        net.revert_move()
        assert np.array_equal(net.gates, backup.gates)


def test_revert_all_moves(any_test_network):
    net = any_test_network['network']
    for i in range(10):
        old_gates = copy(net.gates)
        for i in range(50):
            net.move_to_neighbour(net.random_move())
        net.revert_all_moves()
        assert np.array_equal(net.gates, old_gates)


def test_multiple_move_gates(any_test_network):
    net = any_test_network['network']
    for i in range(100):
        old_gates = copy(net.gates)
        net.move_to_neighbour(net.random_move())
        assert not np.array_equal(net.gates, old_gates)


def test_move_in_connected_range(any_test_network):
    net = any_test_network['network']

    connected = connected_ground_truth(net)

    for i in range(50):
        move = net.random_move()
        assert move[0] in connected


def test_masked_random_move(adder2):
    # the following have no valid sourceable nodes for any of the
    # changeable ones - remember 2 bit adders means sources 0-3 are
    # inputs and only source 4-> are gates
    # this means that ([1,2,3], [0]) is valid since gate 0, is source 4
    net = adder2['network']
    connected = connected_ground_truth(net)

    gates = net.gates.tolist()

    valid_cases = [(gates[3], [3]), ([1,  2, 3], [0]), ([1, 2, 5], [7, 8, 9]),
                   (list(range(6)), list(range(9, len(gates)))),
                   (list(range(6)), list(range(3, 9))),
                   (list(range(6)), list(range(3, len(gates)))),
                   (list(range(len(gates))), list(range(len(gates))))]

    for sourceable, changeable in valid_cases:
        for i in range(50):
            # get a random restricted move
            net.set_mask(sourceable, changeable)
            gate, terminal, new_source = net.random_move()
            # check the move is changing a valid gate
            assert gate in changeable
            assert gate in connected
            # check the move is connecting to a valid source
            assert new_source in sourceable
            # check the move has changed the original connection
            assert new_source != gates[gate][terminal]


def test_reconnect_masked_range(adder2):
    # the following have no valid sourceable nodes for any of the
    # changeable ones - remember 2 bit adders means sources 0-3 are
    # inputs and only source 4-> are gates
    # this means that ([1,2,3], [0]) is valid since gate 0, is source 4
    net = adder2['network']

    gates = net.gates.tolist()

    valid_cases = [(gates[3], [3]), ([1,  2, 3], [0]), ([1, 2, 5], [7, 8, 9]),
                   (list(range(6)), list(range(9, len(gates)))),
                   (list(range(6)), list(range(3, 9))),
                   (list(range(6)), list(range(3, len(gates)))),
                   (list(range(len(gates))), list(range(len(gates))))]

    for sourceable, changeable in valid_cases:
        for i in range(50):
            # get a random restricted move
            net.set_mask(sourceable, changeable)
            net.reconnect_masked_range()
            # check feedforwardness
            for g in range(net.Ng):
                assert net.gates[g][0] < g + net.Ni
                assert net.gates[g][1] < g + net.Ni
            valid = set(sourceable).union(g+net.Ni for g in changeable)
            # check the changeable range all connect to sourceable nodes
            for g in changeable:
                assert net.gates[g][0] in valid
                assert net.gates[g][1] in valid


def test_max_depth(any_test_network):
    net = any_test_network['network']

    actual = net.max_node_depths()
    expected = any_test_network['max depths']

    assert np.array_equal(actual, expected)
