cimport numpy as np

cdef size_t sample_mask(np.uint8_t[:] mask, size_t max_index=*)

cpdef connected_sources(np.uint32_t[:, :] gates, np.uint8_t[:] connected,
                        size_t Ni, size_t No)