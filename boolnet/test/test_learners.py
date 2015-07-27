import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from boolnet.network.boolnetwork import BoolNetwork
from boolnet.learning.learners import strata_boundaries, get_mask


# #################### Global fixtures #################### #
@fixture(params=[
    (100, 16, 8, [0, 11, 23, 34, 46, 57, 69, 80, 92]),
    (173, 4, 9, [0, 18, 36, 54, 72, 91, 109, 127, 145, 164])])
def harness(request):
    # load experiment file
    Ng, Ni, No, bounds = request.param
    gates = np.empty(shape=(Ng, 2), dtype=np.int32)
    for g in range(Ng):
        gates[g, :] = np.random.randint(g+Ni, size=2)
    net = BoolNetwork(gates, Ni, No)
    lower = bounds[:-1]
    upper = bounds[1:]
    return net, lower, upper


# @fixture(params=[
#     (100, 16, 8, [0, 11, 23, 34, 46, 57, 69, 80, 92]),
#     (173, 4, 9, [0, 18, 36, 54, 72, 91, 109, 127, 145, 164])])
# def kfs_harness(request):
#     # load experiment file
#     Ng, Ni, No, bounds = request.param
#     gates = np.empty(shape=(Ng, 2), dtype=np.int32)
#     for g in range(Ng):
#         gates[g, :] = np.random.randint(g+Ni, size=2)
#     net = BoolNetwork(gates, Ni, No)
#     lower = bounds[:-1]
#     upper = bounds[1:]

#     return net, lower, upper, feature_sets


def test_strata_boundaries_overlap(harness):
    network, _, _ = harness
    lower, upper = strata_boundaries(network)
    assert lower[1:] == upper[:-1]


def test_strata_boundaries_even(harness):
    network, _, _ = harness
    lower, upper = strata_boundaries(network)
    diff = [u - l for l, u in zip(lower, upper)]
    assert 0 <= max(diff) - min(diff) <= 1


def test_strata_boundaries_lower(harness):
    network, expected, _ = harness
    actual, _ = strata_boundaries(network)
    assert expected == actual


def test_strata_boundaries_upper(harness):
    network, _, expected = harness
    _, actual = strata_boundaries(network)
    assert expected == actual


def test_get_mask_without_kfs_sourceable(harness):
    net, lower, upper = harness

    Ni, No, Ng = net.Ni, net.No, net.Ng

    for tgt, (l, u) in enumerate(zip(lower, upper)):
        expected = np.array([1]*(Ni+l) + [1]*(u-l) + [0]*(Ng-No-u) +
                            [1]*tgt + [0]*(No-tgt), dtype=np.uint8)
        actual, _ = get_mask(net, l, u, tgt)
        print(expected)
        print(actual)
        assert_array_equal(expected, actual)


def test_get_mask_without_kfs_changeable(harness):
    net, lower, upper = harness

    No, Ng = net.No, net.Ng

    for tgt, (l, u) in enumerate(zip(lower, upper)):
        expected = np.array([0]*l + [1]*(u-l) + [0]*(Ng-No-u) +
                            [0]*tgt + [1] + [0]*(No-tgt-1), dtype=np.uint8)
        _, actual = get_mask(net, l, u, tgt)
        print(expected)
        print(actual)
        assert_array_equal(expected, actual)


# def test_get_mask_with_kfs_sourceable(kfs_harness):
#     net, lower, upper, feature_sets = kfs_harness

#     Ni, No, Ng = net.Ni, net.No, net.Ng

#     for tgt, (l, u) in enumerate(zip(lower, upper)):
#         expected = np.array([1]*(Ni+l) + [1]*(u-l) + [0]*(Ng-No-u) +
#                             [1]*tgt + [0]*(No-tgt), dtype=np.uint8)
#         actual, _ = get_mask(net, l, u, tgt)
#         print(expected)
#         print(actual)
#         assert_array_equal(expected, actual)


# def test_get_mask_with_kfs_changeable(kfs_harness):
#     net, lower, upper, feature_sets = kfs_harness

#     No, Ng = net.No, net.Ng

#     for tgt, (l, u) in enumerate(zip(lower, upper)):
#         expected = np.array([0]*l + [1]*(u-l) + [0]*(Ng-No-u) +
#                             [0]*tgt + [1] + [0]*(No-tgt-1), dtype=np.uint8)
#         _, actual = get_mask(net, l, u, tgt)
#         print(expected)
#         print(actual)
#         assert_array_equal(expected, actual)
