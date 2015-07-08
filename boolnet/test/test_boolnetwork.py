import yaml
import glob
import networkx as nx
import numpy as np
from numpy.random import randint
from copy import copy, deepcopy
from itertools import chain
from boolnet.network.boolnetwork import BoolNetwork, RandomBoolNetwork
from pytest import fixture, raises
from pytest import mark


# ############# Global helpers ################# #
def load_field_from_yaml(filename, fieldname):
    with open(filename) as f:
        d = yaml.safe_load(f)
    return d[fieldname]


def harness_to_fixture(stream):
    test = yaml.safe_load(stream)
    Ni = test['Ni']
    No = test['No']
    gates = np.array(test['gates'], np.uint32)
    # add network to test
    if 'transfer functions' in test:
        tf = test['transfer functions']
        test['net'] = RandomBoolNetwork(gates, Ni, No, tf)
    else:
        test['net'] = BoolNetwork(gates, Ni, No)
    return test


# #################### Global fixtures #################### #
@fixture(scope='module')
def adder2():
    with open('boolnet/test/networks/adder2.yaml') as f:
        net = harness_to_fixture(f)
    return net['net']


@fixture(
    scope='module',
    params=load_field_from_yaml('boolnet/test/networks/adder2.yaml', 'valid_masks'))
def adder2_valid_mask(request):
    return request.param


@fixture(
    scope='module',
    params=load_field_from_yaml('boolnet/test/networks/adder2.yaml', 'invalid_masks'))
def adder2_invalid_mask(request):
    return request.param


@fixture(
    scope='module',
    params=glob.glob('boolnet/test/networks/*.yaml'))
def network_file_instance(request):
    return request.param


# #################### Exception Testing #################### #
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

    # ############## Fixtures ##################### #
    @fixture
    def network(self):
        return BoolNetwork([(0, 1)], 2, 1)

    @fixture(params=invalid_constructions)
    def invalid_construction(self, request):
        return request.param

    # ############## Tests ##################### #
    def test_construction_exceptions(self, invalid_construction):
        with raises(ValueError):
            BoolNetwork(*invalid_construction)

    def test_set_mask_exceptions(self, adder2, adder2_invalid_mask):
        with raises(ValueError):
            adder2.set_mask(*adder2_invalid_mask)

    def test_initial_revert_move_exception(self, network):
        with raises(RuntimeError):
            network.revert_move()

    def test_1_move_2_revert_exception(self, network):
        move = network.random_move()
        network.move_to_neighbour(move)
        network.revert_move()
        with raises(RuntimeError):
            network.revert_move()

    def test_n_move_n_plus_2_revert_exception(self, network):
        # n moves, n + 2 reverts
        for i in range(10):
            network.move_to_neighbour(network.random_move())
        for i in range(10):
            network.revert_move()
        with raises(RuntimeError):
            network.revert_move()
        with raises(RuntimeError):
            network.revert_move()

    def test_revert_after_revert_all_exception(self, network):
        # revert after revert all
        for i in range(10):
            network.move_to_neighbour(network.random_move())
        network.revert_all_moves()
        with raises(RuntimeError):
            network.revert_move()


# #################### Functional Testing #################### #
class TestFunctionalilty:
    # ############## Helper Methods ##################### #
    def graph_from_gates(self, gates, Ni, No):
        DG = nx.DiGraph()
        DG.add_nodes_from(range(Ni))
        edges = []
        for g in range(len(gates)):
            edges.append((gates[g][0], g+Ni))
            edges.append((gates[g][1], g+Ni))
        DG.add_edges_from(edges)
        return DG

    def connected_ground_truth(self, net):
        Ni = net.Ni
        No = net.No
        Ng = net.Ng
        digraph = self.graph_from_gates(net.gates, Ni, No)
        connected = set(chain.from_iterable(
            nx.ancestors(digraph, Ni+Ng-o-1) for o in range(No)))
        connected = np.unique([g - Ni for g in connected if g >= Ni])
        connected = np.union1d(connected, np.arange(Ng-No, Ng))
        return connected

    def assert_feedforward(self, net):
        for g in range(net.Ng):
            assert net.gates[g][0] < g + net.Ni
            assert net.gates[g][1] < g + net.Ni

    # ############## Fixtures ##################### #
    @fixture(scope='module')
    def any_network(self, network_file_instance):
        with open(network_file_instance) as f:
            net = harness_to_fixture(f)
        return net

    @fixture(scope='module')
    def random_network(self):
        # generate random feedforward network
        Ni = randint(1, 10)
        No = randint(1, 10)
        Ng = randint(No, 20)
        random_gates = [(randint(0, g+Ni), randint(0, g+Ni)) for g in range(Ng)]
        # create the seed network
        return BoolNetwork(random_gates, Ni, No)

    # ############## Tests ##################### #
    def test_acyclic(self, random_network):
        DG = self.graph_from_gates(random_network.gates, random_network.Ni, random_network.No)
        assert nx.is_directed_acyclic_graph(DG)

    def test_feedforward(self, random_network):
        self.assert_feedforward(random_network)

    def test_connected_gates(self, random_network):
        expected = self.connected_ground_truth(random_network)
        actual = random_network.connected_gates(random_network.No)
        assert np.array_equal(actual, expected)

    def test_revert_move_revert(self, any_network):
        net = deepcopy(any_network['net'])
        for i in range(20):
            old_gates = copy(net.gates)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(net.gates, old_gates)
            net.revert_move()
            assert np.array_equal(net.gates, old_gates)

    def test_revert_move_revert_multi(self, any_network):
        net = deepcopy(any_network['net'])
        backups = []
        # move network several times keeping a list of the networks
        for i in range(20):
            backups.append(deepcopy(net))
            net.move_to_neighbour(net.random_move())
        # revert it incrementally and check it is equal to the backup each time
        for backup in reversed(backups):
            net.revert_move()
            assert np.array_equal(net.gates, backup.gates)

    def test_revert_all_moves(self, any_network):
        net = deepcopy(any_network['net'])
        for i in range(10):
            old_gates = copy(net.gates)
            for i in range(10):
                net.move_to_neighbour(net.random_move())
            net.revert_all_moves()
            assert np.array_equal(net.gates, old_gates)

    def test_multiple_move_gates(self, any_network):
        net = deepcopy(any_network['net'])
        for i in range(20):
            old_gates = copy(net.gates)
            net.move_to_neighbour(net.random_move())
            assert not np.array_equal(net.gates, old_gates)

    @mark.parametrize('repeats', range(10))
    def test_move_in_connected_range(self, repeats, any_network):
        net = deepcopy(any_network['net'])
        connected = self.connected_ground_truth(net)
        gate, _, _ = net.random_move()
        assert gate in connected

    @mark.parametrize('repeats', range(10))
    def test_masked_random_move(self, repeats, adder2, adder2_valid_mask):
        net = deepcopy(adder2)
        sourceable, changeable = adder2_valid_mask
        connected = self.connected_ground_truth(net)
        # get a random restricted move
        net.set_mask(sourceable, changeable)
        gate, terminal, new_source = net.random_move()
        # check the move is changing a valid gate
        assert gate in changeable
        assert gate in connected
        # check the move is connecting to a valid source
        assert new_source in sourceable
        # check the move has changed the original connection
        assert new_source != net.gates[gate, terminal]

    @mark.parametrize('repeats', range(10))
    def test_reconnect_masked_range(self, repeats, adder2, adder2_valid_mask):
        net = deepcopy(adder2)
        sourceable, changeable = adder2_valid_mask
        net.set_mask(sourceable, changeable)
        net.reconnect_masked_range()
        # check the changeable range all connect to sourceable nodes
        valid = set(sourceable).union(g + net.Ni for g in changeable)
        for g in changeable:
            assert net.gates[g][0] in valid
            assert net.gates[g][1] in valid

    @mark.parametrize('repeats', range(10))
    def test_feedforward_after_reconnect(self, repeats, adder2, adder2_valid_mask):
        net = deepcopy(adder2)
        net.set_mask(*adder2_valid_mask)
        net.reconnect_masked_range()
        self.assert_feedforward(net)

    def test_max_depth(self, any_network):
        net = deepcopy(any_network['net'])
        expected = any_network['max depths']
        actual = net.max_node_depths()
        assert np.array_equal(actual, expected)
