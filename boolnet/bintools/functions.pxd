# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cpdef enum Function:
    E1
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    E1_MCC,
    E2_MCC,
    E6_MCC,
    ACCURACY,
    PER_OUTPUT_ERROR,
    PER_OUTPUT_MCC
    #E1
    #E2M, E2L,
    #E3M, E3L,
    #E4M, E4L,
    #E5M, E5L,
    #E6M, E6L,
    #E7M, E7L,
    #E1_MCC,
    #E2M_MCC, E2L_MCC,
    #E6M_MCC, E6L_MCC,
    #ACCURACY,
    #PER_OUTPUT_ERROR,
    #PER_OUTPUT_MCC
