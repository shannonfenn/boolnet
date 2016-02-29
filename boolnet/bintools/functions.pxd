# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cpdef enum Function:
    E1
    E2M, E2L,    # weighted
    E3M, E3L,    # hierarchical
    E4M, E4L,
    E5M, E5L,
    E6M, E6L,
    E7M, E7L,    # worst example
    ACCURACY,
    PER_OUTPUT,
    MCC