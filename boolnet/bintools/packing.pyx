# cython: language_level=3, profile=False
import numpy as np
cimport numpy as np
from libc.limits cimport ULLONG_MAX


PACKED_SIZE = 64
PACKED_SIZE_PY = PACKED_SIZE
PACKED_ALL_SET = ULLONG_MAX
PACKED_HIGH_BIT_SET = 0x8000000000000000
packed_type = np.uint64


class BitPackedMatrix(np.ndarray):
    def __new__(cls, input_array, Ne, Ni=0):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
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

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(BitPackedMatrix, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.Ne, self.Ni)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.Ne = state[-2]  # Set the info attribute
        self.Ni = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(BitPackedMatrix, self).__setstate__(state[0:-2])


cpdef pack_chunk(packed_type_t[:] mat, packed_type_t[:, :] packed, size_t Nf, size_t column):
    ''' This method assumed mat.shape[0] == PACKED_SIZE.'''
    cdef:
        size_t f, bit
        packed_type_t chunk, mask
    # build packed matrix
    mask = 1
    for f in range(Nf):
        chunk = 0
        for bit in range(PACKED_SIZE):
            chunk |= (((mat[bit] & mask) >> f) << bit)
        mask <<= 1
        packed[f, column] = chunk


cpdef pack_bool_matrix(np.ndarray mat):
    cdef:
        size_t Ne, Nf, num_chunks, f, c, bit
        packed_type_t chunk
        packed_type_t[:, :] packed
        np.ndarray[np.uint8_t, ndim=2] padded

    Ne, Nf = mat.shape[0], mat.shape[1]
    num_chunks = int(np.ceil(Ne / <double>PACKED_SIZE))
    # pad rows with zeros to next multiple of PACKED_SIZE 
    padded = np.zeros((num_chunks * PACKED_SIZE, Nf), dtype=np.uint8)
    padded[:Ne, :] = mat
    # build packed matrix
    packed = np.empty((Nf, num_chunks), dtype=np.uint64)
    for f in range(Nf):
        for c in range(num_chunks):
            chunk = 0
            for bit in range(PACKED_SIZE):
                chunk |= (<packed_type_t>(padded[c*PACKED_SIZE+bit, f]) << bit)
            packed[f, c] = chunk
    return np.asarray(packed)


cpdef unpack_bool_matrix(packed_type_t[:, :] packed_mat, size_t Ne):
    cdef:
        size_t Nf, num_chunks, f, c, bit, example
        packed_type_t mask, chunk
        np.uint8_t[:, :] unpacked

    Nf, num_chunks = packed_mat.shape[0], packed_mat.shape[1]
    unpacked = np.zeros((num_chunks*PACKED_SIZE, Nf), dtype=np.uint8)
    for f in range(Nf):
        example = 0
        for c in range(num_chunks):
            mask = 1
            chunk = packed_mat[f, c]
            for bit in range(PACKED_SIZE):
                unpacked[example, f] += ((chunk & mask) >> bit)
                mask <<= 1
                example += 1
    return np.asarray(unpacked[:Ne, :])


cpdef unpack_bool_vector(packed_type_t[:] packed_vec, size_t Ne):
    cdef:
        size_t num_chunks, c, bit, example
        packed_type_t mask, chunk
        np.uint8_t[:] unpacked

    num_chunks = packed_vec.size
    unpacked = np.zeros(num_chunks*PACKED_SIZE, dtype=np.uint8)
    for c in range(num_chunks):
        # mask = 1
        # example = (c + 1) * PACKED_SIZE - 1
        # chunk = packed_vec[c]
        # for bit in range(PACKED_SIZE):
        #     unpacked[example] += ((chunk & mask) >> bit)
        #     mask <<= 1
        #     example -= 1
        mask = 1
        example = c * PACKED_SIZE
        chunk = packed_vec[c]
        for bit in range(PACKED_SIZE):
            unpacked[example] += ((chunk & mask) >> bit)
            mask <<= 1
            example += 1
    return np.asarray(unpacked[:Ne])


cpdef packed_type_t generate_end_mask(Ne):
    cdef:
        packed_type_t end_mask, shift
        size_t bits
    
    end_mask = PACKED_ALL_SET
    bits_to_remain = Ne % PACKED_SIZE

    if bits_to_remain != 0:
        # shift = 1
        # for b in range(PACKED_SIZE-bits_to_remain):
        #     end_mask -= shift
        #     shift <<= 1
        shift = PACKED_HIGH_BIT_SET
        for b in range(PACKED_SIZE-bits_to_remain):
            end_mask &= ~shift
            shift >>= 1
    return end_mask


cpdef partition_packed(matrix, indices):
    cdef packed_type_t mask
    cdef packed_type_t[:, :] M = matrix # typed view
    cdef size_t Nf, Nw, Ne, f, w, b, b_trg, b_test, w_trg, w_test

    if matrix.dtype != packed_type:
        raise ValueError('sample_packed only accepts np.uint64')

    Ne = indices.shape[0]
    Nf, Nw = matrix.shape

    # find number of words in training and test samples
    Nw_trg = int(np.ceil(Ne / <double>PACKED_SIZE))
    Nw_test = int(np.ceil((matrix.Ne - Ne) / <double>PACKED_SIZE))

    M_trg = BitPackedMatrix(np.zeros((Nf, Nw_trg), dtype=packed_type),
                            Ne=Ne, Ni=matrix.Ni)
    M_test = BitPackedMatrix(np.zeros((Nf, Nw_test), dtype=packed_type),
                             Ne=matrix.Ne - Ne, Ni=matrix.Ni)

    for f in range(Nf):
        # word and bit positions for training and test samples
        b_trg = b_test = 0
        w_trg = w_test = 0
        for w in range(Nw):
            for b in range(PACKED_SIZE):
                # get the bit from the original matrix
                mask = 1 << b
                bit = (M[f, w] & mask) >> b
                # if this bit of this word is in the sample
                if b + w * PACKED_SIZE in indices:
                    # insert into training sample at the next position
                    M_trg[f, w_trg] += bit << b_trg
                    # increment training sample word and bit indices
                    b_trg = (b_trg + 1) % PACKED_SIZE
                    w_trg += b_trg == 0
                else:
                    # insert into test sample at the next position
                    M_test[f, w_test] += bit << b_test
                    # increment test sample word and bit indices
                    b_test = (b_test + 1) % PACKED_SIZE
                    w_test += b_test == 0
    return M_trg, M_test


cpdef sample_packed(matrix, indices, invert=False):
    cdef packed_type_t mask
    cdef packed_type_t[:, :] M = matrix # typed view
    cdef size_t Nf, Nw, Ne, f, w, b, sw, sb

    if matrix.dtype != packed_type:
        raise ValueError('sample_packed only accepts np.uint64')

    Nf, Nw = matrix.shape
    Ne = indices.shape[0]

    if invert:
        # sample matrix
        cols = int(np.ceil((matrix.Ne - Ne) / <double>PACKED_SIZE))
        sample = np.zeros((Nf, cols), dtype=packed_type)
        sample = BitPackedMatrix(sample, Ne=matrix.Ne-Ne, Ni=matrix.Ni)
        
        # word and bit positions for sample
        sb = sw = 0
        for w in range(Nw):
            for b in range(PACKED_SIZE):
                # if this bit of this word is in the sample
                if b + w * PACKED_SIZE not in indices:
                    # get the bit
                    mask = 1 << b
                    for f in range(Nf):
                        bit = (M[f, w] & mask) >> b
                        # insert into the sample at the next position
                        sample[f, sw] += bit << sb
                    # increment sample word and bit indices
                    sb = (sb + 1) % PACKED_SIZE
                    sw += sb == 0
    else:
        Nw_trg = int(np.ceil(Ne / <double>PACKED_SIZE))
        sample = np.zeros((Nf, Nw_trg), dtype=packed_type)
        sample = BitPackedMatrix(sample, Ne=Ne, Ni=matrix.Ni)
        
        # word and bit positions for sample
        sw = sb = 0
        # for each index in sample
        for index in indices:
            # word and bit indices into original matrix
            w = index // PACKED_SIZE
            b = index % PACKED_SIZE
            mask = 1 << b
            for f in range(Nf):
                # get the bit
                bit = (M[f, w] & mask) >> b
                # insert into into the sample at the next position
                sample[f, sw] += (bit << sb)
            # increment word and bit indices
            sb = (sb + 1) % PACKED_SIZE
            sw += sb == 0
    return sample


function_list = [__f0, __f1, __f2, __f3, __f4, __f5, __f6, __f7,
                 __f8, __f9, __f10, __f11, __f12, __f13, __f14, __f15,]

cdef packed_type_t __f0(packed_type_t x, packed_type_t y):  return 0
cdef packed_type_t __f1(packed_type_t x, packed_type_t y):  return ~(x|y)   # NOR
cdef packed_type_t __f2(packed_type_t x, packed_type_t y):  return ~x&y
cdef packed_type_t __f3(packed_type_t x, packed_type_t y):  return ~x
cdef packed_type_t __f4(packed_type_t x, packed_type_t y):  return x&~y
cdef packed_type_t __f5(packed_type_t x, packed_type_t y):  return ~y
cdef packed_type_t __f6(packed_type_t x, packed_type_t y):  return x^y      # XOR
cdef packed_type_t __f7(packed_type_t x, packed_type_t y):  return ~(x&y)   # NAND
cdef packed_type_t __f8(packed_type_t x, packed_type_t y):  return x&y      # AND
cdef packed_type_t __f9(packed_type_t x, packed_type_t y):  return ~(x^y)   # XNOR
cdef packed_type_t __f10(packed_type_t x, packed_type_t y): return y
cdef packed_type_t __f11(packed_type_t x, packed_type_t y): return ~x|y
cdef packed_type_t __f12(packed_type_t x, packed_type_t y): return x
cdef packed_type_t __f13(packed_type_t x, packed_type_t y): return x|~y
cdef packed_type_t __f14(packed_type_t x, packed_type_t y): return x|y      # OR
cdef packed_type_t __f15(packed_type_t x, packed_type_t y): return 1