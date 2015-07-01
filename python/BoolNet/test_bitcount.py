from BoolNet.bitcount import floodcount_vector, floodcount_chunk, popcount_vector, popcount_chunk
import numpy as np
from pytest import fixture


@fixture(params=[
    (0, 0, 0),
    (4370323815288199633, 1, 33), (1129843337900893199, 1, 30),
    (649369928250959927, 1, 23), (588858933541175576, 4, 17),
    (1971513181041067719, 1, 32), (3555176812501657937, 1, 33),
    (4558881714062980293, 1, 37), (7796458688392721668, 3, 33),
    (5717748707411915565, 1, 33), (9086724868667118591, 1, 35)] + [
    (2**i, i + 1, 1) for i in range(64)])
def chunkinstance(self, request):
    return request.param


@fixture(params=[
    ([0, 0], 0, 0), ([0, 0, 0], 0, 0), ([0, 0, 0, 0, 0], 0, 0),
    ([4370323815288199633, 1129843337900893199], 1, 63),
    ([588858933541175576, 649369928250959927, ], 4, 40),
    ([1971513181041067719, 3555176812501657937, 7796458688392721668], 3, 98),
    ([4558881714062980293, 9086724868667118591, 5717748707411915565], 1, 105)] + [
    ([2**i, 2**j], i + 1, 2) for i in range(0, 64, 16) for j in range(0, 64, 16)] + [
    ([2**i, 2**j, 2**j], i + 1, 3) for i in range(0, 64, 16) for j in range(0, 64, 16)])
def vectorinstance(self, request):
    return request.param


def test_floodcount_chunk(chunkinstance):
    expected = chunkinstance[1]
    actual = floodcount_chunk(chunkinstance[0])
    assert expected == actual


def test_popcount_chunk(chunkinstance):
    expected = chunkinstance[2]
    actual = popcount_chunk(chunkinstance[0])
    assert expected == actual


def test_floodcount_vector(vectorinstance):
    expected = vectorinstance[1]
    actual = floodcount_vector(np.array(vectorinstance[0], dtype=np.uint64))
    assert expected == actual


def test_popcount_vector(vectorinstance):
    expected = vectorinstance[2]
    actual = popcount_vector(np.array(vectorinstance[0], dtype=np.uint64))
    assert expected == actual
