import numpy as np
from numpy.testing import assert_array_equal
from pytest import fixture
from boolnet.network.boolnetwork import BoolNetwork
from boolnet.learning.learners import build_mask


# #################### Global fixtures #################### #
@fixture(params=[
    (100, 16, 8, [0, 11, 23, 34, 46, 57, 69, 80, 92]),
    (173, 4, 9, [0, 18, 36, 54, 72, 91, 109, 127, 145, 164])])
def harness(request):
    # load experiment file
    Ng, Ni, No, bounds = request.param
    gates = np.empty(shape=(Ng, 3), dtype=np.int32)
    for g in range(Ng):
        gates[g, :-1] = np.random.randint(g+Ni, size=2)
        gates[g, -1] = np.random.randint(16)
    net = BoolNetwork(gates, Ni, No)
    return net, bounds


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
#     return net, bounds, feature_sets


def test_build_mask_without_kfs_sourceable(harness):
    net, bounds = harness

    Ni, No, Ng = net.Ni, net.No, net.Ng
    for target in range(No):
        l, u = bounds[target], bounds[target + 1]
        # commented out: version where prior outputs were sourceable
        # expected = np.array([1]*(Ni+l) + [1]*(u-l) + [0]*(Ng-No-u) +
        #                     [1]*target + [0]*(No-target), dtype=np.uint8)
        expected = np.array([1]*(Ni+l) + [1]*(u-l) + [0]*(Ng-No-u) +
                            [0]*(No), dtype=np.uint8)
        actual, _ = build_mask(net, l, u, target)
        print('expected:\n', expected)
        print('actual:\n', actual)
        assert_array_equal(expected, actual)


def test_build_mask_without_kfs_changeable(harness):
    net, bounds = harness

    No, Ng = net.No, net.Ng

    for target in range(No):
        l, u = bounds[target], bounds[target + 1]
        expected = np.array([0]*l + [1]*(u-l) + [0]*(Ng-No-u) +
                            [0]*target + [1] + [0]*(No-target-1),
                            dtype=np.uint8)
        _, actual = build_mask(net, l, u, target)
        # print('expected:\n', expected)
        # print('actual:\n', actual)
        assert_array_equal(expected, actual)


# def test_build_mask_with_kfs_sourceable(kfs_harness):
#     net, bounds, feature_sets = kfs_harness

#     Ni, No, Ng = net.Ni, net.No, net.Ng

#     for target in range(No):
#         l, u = bounds[target], bounds[target + 1]
#         expected = np.array([1]*(Ni+l) + [1]*(u-l) + [0]*(Ng-No-u) +
#                             [1]*target + [0]*(No-target), dtype=np.uint8)
#         actual, _ = build_mask(net, l, u, target)
#         print(expected)
#         print(actual)
#         assert_array_equal(expected, actual)


# def test_build_mask_with_kfs_changeable(kfs_harness):
#     net, bounds, feature_sets = kfs_harness

#     No, Ng = net.No, net.Ng

#     for target in range(No):
#         l, u = bounds[target], bounds[target + 1]
#         expected = np.array([0]*l + [1]*(u-l) + [0]*(Ng-No-u) +
#                             [0]*target + [1] + [0]*(No-target-1), dtype=np.uint8)
#         _, actual = build_mask(net, l, u, target)
#         print(expected)
#         print(actual)
#         assert_array_equal(expected, actual)
