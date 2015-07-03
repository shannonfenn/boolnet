# cython: language_level=3, profile=False
import numpy as np
cimport numpy as np
from libc.limits cimport ULLONG_MAX


PACKED_SIZE = 64
PACKED_ALL_SET = ULLONG_MAX
PACKED_HIGH_BIT_SET = 0x8000000000000000
packed_type = np.uint64


cpdef pack_chunk(packed_type_t[:] mat, packed_type_t[:, :] packed, size_t Nf, size_t column):
    ''' This method assumed mat.shape[0] >= 64.'''
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