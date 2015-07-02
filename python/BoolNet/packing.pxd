# cython: language_level=3
import numpy as np
cimport numpy as np


ctypedef np.uint64_t packed_type_t

# cdef enum:
#     PACKED_SIZE = 64
#DEF PACKED_SIZE = 64
cpdef size_t PACKED_SIZE
cpdef packed_type_t PACKED_ALL_SET
cpdef packed_type_t PACKED_HIGH_BIT_SET


cpdef pack_bool_matrix(np.ndarray mat)
cpdef unpack_bool_matrix(packed_type_t[:, :] packed_mat, size_t Ne)
cpdef unpack_bool_vector(packed_type_t[:] packed_vec, size_t Ne)
cpdef packed_type_t generate_end_mask(Ne)

ctypedef packed_type_t (*f_type)(packed_type_t, packed_type_t)
cdef f_type* function_list