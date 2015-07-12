import numpy as np
cimport numpy as np
cimport boolnet.network.algorithms as algorithms

def test():
    cdef np.uint8_t[:] mask1 = np.array(np.random.randint(2, size=15), dtype=np.uint8)
    cdef np.uint8_t[:] mask2 = np.array(np.random.randint(2, size=15), dtype=np.uint8)

    counter1 = [0]*15
    counter2 = [0]*15
    counter3 = [0]*15
    counter4 = [0]*15
    counter5 = [0]*15
    counter6 = [0]*15

    limit1 = np.random.randint(np.sum(mask1))
    limit2 = np.random.randint(np.sum(mask2))
    limit1 = np.flatnonzero(np.asarray(mask1))[limit1]
    limit2 = np.flatnonzero(np.asarray(mask2))[limit2]
    limit3 = max(limit1, limit2)

    print('limits: {} {}'.format(limit1, limit2))

    for i in range(100000):
        v1 = algorithms.sample_bool(mask1)
        v2 = algorithms.sample_bool(mask2)
        v3 = algorithms.sample_bool(mask1, limit1)
        v4 = algorithms.sample_bool(mask2, limit2)
        v5 = algorithms.sample_masked_bool(mask1, mask2)
        v6 = algorithms.sample_masked_bool(mask1, mask2, limit3)
        counter1[v1] += 1
        counter2[v2] += 1
        counter3[v3] += 1
        counter4[v4] += 1
        counter5[v5] += 1
        counter6[v6] += 1

    print('sample mask 1')
    for i in range(15):
        print('{}\t{}\t{}'.format(i, mask1[i], counter1[i]))
    print('sample mask 2')
    for i in range(15):
        print('{}\t{}\t{}'.format(i, mask2[i], counter2[i]))

    print('sample mask 1 with limit: {}'.format(limit1))
    for i in range(15):
        print('{}\t{}\t{}'.format(i, mask1[i], counter3[i]))
    print('sample mask 2 with limit: {}'.format(limit2))
    for i in range(15):
        print('{}\t{}\t{}'.format(i, mask2[i], counter4[i]))

    print('sample mask1 & mask2')
    for i in range(15):
        print('{}\t{}\t{}'.format(i, mask1[i] & mask2[i], counter5[i]))
    print('sample mask1 & mask2 with limit: {}'.format(limit3))
    for i in range(15):
        print('{}\t{}\t{}'.format(i, mask1[i] & mask2[i], counter6[i]))
