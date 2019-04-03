import numpy as np
import bitpacking.packing as pk
from progress.bar import IncrementalBar
import json
from itertools import tee


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BetterETABar(IncrementalBar):
    suffix = ('%(index)d/%(max)d | elapsed: %(elapsed)ds | '
              'eta: %(better_eta)ds')

    @property
    def better_eta(self):
        return self.elapsed / (self.index + 1) * self.remaining

    def writeln(self, line):
        if self.file.isatty():
            self.clearln()
            print('\x1b[?7l' + line + '\x1b[?7h', end='', file=self.file)
            self.file.flush()


class PackedMatrix(np.ndarray):
    def __new__(cls, matrix, Ne, Ni=0):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(matrix).view(cls)
        # add the new attribute to the created instance
        obj.Ne = Ne
        obj.Ni = Ni
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.Ne = getattr(obj, 'Ne', None)
        self.Ni = getattr(obj, 'Ni', None)

    @property
    def No(self):
        return self.shape[0] - self.Ni

    def split(self):
        return np.split(self, [self.Ni])

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(PackedMatrix, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.Ne, self.Ni)
        # Return a tuple that replaces the parent's __setstate__ with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.Ne = state[-2]  # Set the info attribute
        self.Ni = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(PackedMatrix, self).__setstate__(state[0:-2])


def partition_packed(matrix, indices):
    unpacked_matrix = unpack(matrix)
    M_in = pack(unpacked_matrix[indices, :], Ni=matrix.Ni)
    M_out = pack(np.delete(unpacked_matrix, indices, axis=0), Ni=matrix.Ni)
    return M_in, M_out


def sample_packed(matrix, indices, invert=False):
    unpacked_matrix = unpack(matrix)
    if invert:
        return pack(np.delete(unpacked_matrix, indices, axis=0), Ni=matrix.Ni)
    else:
        return pack(unpacked_matrix[indices, :], Ni=matrix.Ni)


def unpack(packed_matrix, transpose=True):
    return pk.unpackmat(packed_matrix, packed_matrix.Ne, transpose=transpose)


def pack(unpacked_matrix, Ni=0, transpose=True):
    Ne = unpacked_matrix.shape[0] if transpose else unpacked_matrix.shape[1]
    packed_matrix = pk.packmat(unpacked_matrix, transpose=transpose)
    return PackedMatrix(packed_matrix, Ne=Ne, Ni=Ni)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def spacings(n, k, low_first=True):
    # (n + k - 1) // k == ceil(n/k)
    S = (k - n % k) * [n // k] + (n % k) * [(n + k - 1) // k]
    if not low_first:
        S.reverse()
    return S
