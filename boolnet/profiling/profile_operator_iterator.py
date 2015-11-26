from collections import deque
import boolnet.bintools.operator_iterator as opit
import numpy as np
import timeit
import shlex

INCLUDE_ITERS = {
    'Zero': opit.ZeroIncludeIterator,
    'uAND': opit.UnaryANDIncludeIterator,
    'uOR': opit.UnaryORIncludeIterator,
    'bAND': opit.ANDIncludeIterator,
    'bOR': opit.ORIncludeIterator,
    'Add': opit.AddIncludeIterator,
    'Sub': opit.SubIncludeIterator,
    'Mul': opit.MulIncludeIterator
}

EXCLUDE_ITERS = {
    'Zero': opit.ZeroExcludeIterator,
    'uAND': opit.UnaryANDExcludeIterator,
    'uOR': opit.UnaryORExcludeIterator,
    'bAND': opit.ANDExcludeIterator,
    'bOR': opit.ORExcludeIterator,
    'Add': opit.AddExcludeIterator,
    'Sub': opit.SubExcludeIterator,
    'Mul': opit.MulExcludeIterator
}

INC_LISTS = {
    4: np.load('samples4.npy'),
    8: np.load('samples8.npy')
}

exhaust = deque(maxlen=0).extend

print('Include-style iterators')
for Nb in sorted(INC_LISTS):
    inc = INC_LISTS[Nb]
    print('Nb:', Nb, 'Ne:', len(inc))
    for name in sorted(INCLUDE_ITERS):
        itr_class = INCLUDE_ITERS[name]
        print(name, end="\t")

        setup = 'from __main__ import itr_class, inc, Nb, exhaust; '
        # setup += 'itr = itr_class(inc, Nb)'
        stmnt = 'itr = itr_class(inc, Nb); exhaust(itr)'
        # flags = ' -n 1000000'  # ' -r 1'
        flags = ''  # ' -r 1'

        cmd = '-s\'{}\' {} \'{}\''.format(setup, flags, stmnt)

        timeit.main(args=shlex.split(cmd))

print('Exclude-style iterators')
for Nb in sorted(INC_LISTS):
    inc = INC_LISTS[Nb]
    max_elems = 2**(2*Nb)
    print('Nb:', Nb, 'Ne:', max_elems - len(inc))
    for name in sorted(EXCLUDE_ITERS):
        itr_class = EXCLUDE_ITERS[name]
        print(name, end="\t")

        setup = 'from __main__ import itr_class, inc, Nb, max_elems, exhaust;'
        # setup += 'itr = itr_class(inc, Nb, max_elems)'
        stmnt = 'itr = itr_class(inc, Nb, max_elems); exhaust(itr)'
        flags = ''  # ' -r 1'

        cmd = '-s\'{}\' {} \'{}\''.format(setup, flags, stmnt)

        timeit.main(args=shlex.split(cmd))
