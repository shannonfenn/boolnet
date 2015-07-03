import numpy as np
import line_profiler
import timeit
import shlex
import argparse
from BoolNet.Packing import pack_bool_matrix, unpack_bool_matrix, unpack_bool_vector
from BoolNet.BooleanNetwork import BooleanNetwork


def dump_stats(profile, func_name):
    profile.print_stats()
    stats = profile.get_stats()
    assert len(stats.timings) > 0, "No profile stats."
    for key, timings in stats.timings.items():
        if key[-1] == 'unpack_bool_matrix':
            assert len(timings) > 0
            break
    else:
        raise ValueError("No stats for {}.".format(func_name))


def profile_packing(line=False):
    # setup
    A = np.array(np.random.randint(2, size=(32, 50)), dtype=np.uint8)
    Ap = pack_bool_matrix(A)
    Vp = Ap[0, :]

    timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

    print('BASIC TIMING')
    print('pack_bool_matrix')
    timeit.main(args=shlex.split(
        """-s'from __main__ import A, pack_bool_matrix' 'pack_bool_matrix(A)'"""))
    print('unpack_bool_vector')
    timeit.main(args=shlex.split(
        """-s'from __main__ import Vp, unpack_bool_vector' 'unpack_bool_vector(Vp, 32)'"""))
    print('unpack_bool_matrix')
    timeit.main(args=shlex.split(
        """-s'from __main__ import Ap, unpack_bool_matrix' 'unpack_bool_matrix(Ap, 32)'"""))

    if line:
        print('\nLINE PROFILING')
        profile = line_profiler.LineProfiler(pack_bool_matrix)
        profile.runcall(pack_bool_matrix, A)
        dump_stats(profile)
        profile = line_profiler.LineProfiler(unpack_bool_matrix)
        profile.runcall(unpack_bool_matrix, Ap, 32)
        dump_stats(profile)
        profile = line_profiler.LineProfiler(unpack_bool_vector)
        profile.runcall(unpack_bool_vector, Vp, 32)
        dump_stats(profile)


def random_network():
    Ng = 150
    Ni = 16
    No = 8
    gates = np.empty(shape=(Ng, 2), dtype=np.int32)
    for g in range(Ng):
        gates[g, :] = np.random.randint(g+Ni, size=2)
    return BooleanNetwork(gates, Ni, No)


def profile_BooleanNetwork():
    print('BASIC TIMING')
    print('set_mask')
    cmd = '-s\'from __main__ import random_network, np;'
    cmd += ' net = random_network();'
    cmd += ' sourceable = np.random.choice(np.arange(50), size=25);'
    cmd += ' changeable = np.arange(50, 100)\''
    cmd += ' \'net.set_mask(sourceable, changeable)\''
    timeit.main(args=shlex.split(cmd))

    net = random_network()
    sourceable = np.random.choice(np.arange(50), size=25)
    changeable = np.arange(50, 100)
    net.set_mask(sourceable, changeable)

    print('random_move')
    cmd = '-s\'from __main__ import random_network; net = random_network()\''
    cmd += ' \'net.random_move()\''
    timeit.main(args=shlex.split(cmd))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('module',
                        choices=['packing', 'network'])
    parser.add_argument('--line', action='store_true')

    args = parser.parse_args()

    if args.module == 'packing':
        profile_packing(args.line)
    else:
        profile_BooleanNetwork()
