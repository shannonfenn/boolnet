import pytest
import boolnet.utils as utils
from random import randint


@pytest.mark.parametrize('execution_number', range(10))
def test_spacings(execution_number):
    n = randint(1, 100)
    k = randint(1, n)
    S = utils.spacings(n, k)
    assert sum(S) == n
    assert len(S) == k
    assert all((n // k) <= s <= (n // k + 1) for s in S)
