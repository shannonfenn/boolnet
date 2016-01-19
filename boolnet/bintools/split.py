import numpy as np
from boolnet.bintools.packing import BitPackedArray


def split_packed_mapping(D, sample):
    word_size = D.dtype.itemsize * 8
    Ne = sample.shape[0]
    Nf, Nw = D.shape
    Nw_trg = int(np.ceil(Ne / word_size))
    Nw_test = int(np.ceil((D.N - Ne) / word_size))

    D_trg = BitPackedArray(np.zeros((Nf, Nw_trg), dtype=D.dtype))
    D_trg.N = Ne
    D_test = BitPackedArray(np.zeros((Nf, Nw_test), dtype=D.dtype))
    D_test.N = D.N - Ne

    for f in range(Nf):
        i_trg = i_test = 0
        w_trg = w_test = 0
        for w in range(Nw):
            for i in range(word_size):
                bit = (D & (1 << i)) >> i
                if i + w * word_size in sample:
                    D_trg[w_trg] += bit << i_trg
                    i_trg += 1
                    if i_trg == word_size:
                        i_trg = 0
                        w_trg += 1
                else:
                    D_test[w_test] += bit << i_test
                    i_test += 1
                    if i_test == word_size:
                        i_test = 0
                        w_test += 1
    return D_trg, D_test
