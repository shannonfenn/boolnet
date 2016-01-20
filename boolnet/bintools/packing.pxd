# cython: language_level=3
import numpy as np
cimport numpy as np


ctypedef np.uint64_t packed_type_t

cdef size_t PACKED_SIZE
cdef packed_type_t PACKED_ALL_SET
cdef packed_type_t PACKED_HIGH_BIT_SET


cpdef pack_chunk(packed_type_t[:] mat, packed_type_t[:, :] packed, size_t Nf, size_t column)
cpdef pack_bool_matrix(np.ndarray mat)

cpdef unpack_bool_matrix(packed_type_t[:, :] packed_mat, size_t Ne)
cpdef unpack_bool_vector(packed_type_t[:] packed_vec, size_t Ne)

cpdef packed_type_t generate_end_mask(Ne)

cpdef sample_packed(D, indices, invert=*)

ctypedef packed_type_t (*f_type)(packed_type_t, packed_type_t)

cdef f_type* function_list