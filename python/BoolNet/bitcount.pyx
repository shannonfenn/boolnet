# distutils: libraries = gmp
from Packing cimport PACKED_HIGH_BIT_SET, PACKED_SIZE

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
cpdef size_t popcount_matrix(packed_type_t[:, :] mat):
    return mpn_popcount(&mat[0, 0], mat.size)


cpdef size_t popcount_vector(packed_type_t[:] vec):
    return mpn_popcount(&vec[0], vec.size)


cpdef size_t popcount_chunk(packed_type_t chunk):
    return mpn_popcount(&chunk, 1)


cpdef size_t floodcount_vector(packed_type_t[:] vec):
    cdef bint was_zero
    cdef size_t pos

    # to ensure there is at least a 1 at the end for mpn_scan1
    # we need to add it, but also remember if the last bit was
    # set so we can unset it, and also return the correct count
    was_zero = ((vec[-1] & PACKED_HIGH_BIT_SET) == 0)
    vec[-1] |= PACKED_HIGH_BIT_SET
    pos = mpn_scan1(&vec[0], 0)
    if was_zero:
        vec[-1] &= ~PACKED_HIGH_BIT_SET
        if pos == PACKED_SIZE - 1:
            return 0
    return pos + 1


cpdef size_t floodcount_chunk(packed_type_t chunk):
    cdef bint was_zero
    cdef size_t pos

    # to ensure there is at least a 1 at the end for mpn_scan1
    # we need to add it, but also remember if the last bit was
    # set so we can unset it, and also return the correct count
    was_zero = ((chunk & PACKED_HIGH_BIT_SET) == 0)
    chunk |= PACKED_HIGH_BIT_SET
    pos = mpn_scan1(&chunk, 0)
    if was_zero:
        chunk &= ~PACKED_HIGH_BIT_SET
        if pos == PACKED_SIZE - 1:
            return 0
    return pos + 1
