# cython: profile=False
# distutils: libraries = gmp

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
cpdef size_t popcount_matrix(packed_type_t[:, :] mat):
    cdef size_t n = mat.size
    cdef packed_type_t* data_ptr = &mat[0, 0]
    return mpn_popcount(data_ptr, n)


cpdef size_t popcount_vector(packed_type_t[:] vec):
    cdef size_t n = vec.size
    cdef packed_type_t* data_ptr = &vec[0]
    return mpn_popcount(data_ptr, n)


cpdef size_t popcount_chunk(packed_type_t chunk):
    return mpn_popcount(&chunk, 1)
# # to get a pointer
# cpdef test_pointer(data_type_t[:] view):
#     cdef Py_ssize_t i, j, n = view.shape[0]
#     cdef data_type_t * data_ptr = &view[0]
