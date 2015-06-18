# distutils: language = c++
# distutils: sources = BitError.cpp
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args = -std=c++11


cdef extern from "BitError.hpp" namespace "BitError":
    cpdef enum Metric:
        SIMPLE,
        WEIGHTED_LIN_MSB,        WEIGHTED_LIN_LSB,
        WEIGHTED_EXP_MSB,        WEIGHTED_EXP_LSB,
        HIERARCHICAL_LIN_MSB,    HIERARCHICAL_LIN_LSB,
        HIERARCHICAL_EXP_MSB,    HIERARCHICAL_EXP_LSB,
        WORST_SAMPLE_LIN_MSB,    WORST_SAMPLE_LIN_LSB,
        WORST_SAMPLE_EXP_MSB,    WORST_SAMPLE_EXP_LSB,
        E4_MSB,                  E4_LSB,
        E5_MSB,                  E5_LSB,
        E6_MSB,                  E6_LSB,
        E7_MSB,                  E7_LSB

__METRIC_NAME_MAP = {
     'simple'               : SIMPLE,
     'weighted_lin_msb'     : WEIGHTED_LIN_MSB,        
     'weighted_lin_lsb'     : WEIGHTED_LIN_LSB,        
     'weighted_exp_msb'     : WEIGHTED_EXP_MSB,        
     'weighted_exp_lsb'     : WEIGHTED_EXP_LSB,        
     'hierarchical_lin_msb' : HIERARCHICAL_LIN_MSB,    
     'hierarchical_lin_lsb' : HIERARCHICAL_LIN_LSB,        
     'hierarchical_exp_msb' : HIERARCHICAL_EXP_MSB,    
     'hierarchical_exp_lsb' : HIERARCHICAL_EXP_LSB,        
     'worst_sample_lin_msb' : WORST_SAMPLE_LIN_MSB,    
     'worst_sample_lin_lsb' : WORST_SAMPLE_LIN_LSB,        
     'worst_sample_exp_msb' : WORST_SAMPLE_EXP_MSB,    
     'worst_sample_exp_lsb' : WORST_SAMPLE_EXP_LSB,
     'e4_msb'               : E4_MSB,                  
     'e4_lsb'               : E4_LSB,
     'e5_msb'               : E5_MSB,                  
     'e5_lsb'               : E5_LSB,
     'e6_msb'               : E6_MSB,                  
     'e6_lsb'               : E6_LSB,
     'e7_msb'               : E7_MSB,                  
     'e7_lsb'               : E7_LSB}

__INV_METRIC_NAME_MAP = { v: k for k, v in __METRIC_NAME_MAP.items() }

def metric_from_name(name):
    return __METRIC_NAME_MAP[name]

def metric_name(metric):
    return __INV_METRIC_NAME_MAP[metric]

# cdef class pyMETRIC:
#     METRIC_SIMPLE = SIMPLE
#     METRIC_WEIGHTED_LIN_MSB = WEIGHTED_LIN_MSB        
#     METRIC_WEIGHTED_LIN_LSB = WEIGHTED_LIN_LSB        
#     METRIC_WEIGHTED_EXP_MSB = WEIGHTED_EXP_MSB        
#     METRIC_WEIGHTED_EXP_LSB = WEIGHTED_EXP_LSB        
#     METRIC_HIERARCHICAL_LIN_MSB = HIERARCHICAL_LIN_MSB    
#     METRIC_HIERARCHICAL_LIN_LSB = HIERARCHICAL_LIN_LSB        
#     METRIC_HIERARCHICAL_EXP_MSB = HIERARCHICAL_EXP_MSB    
#     METRIC_HIERARCHICAL_EXP_LSB = HIERARCHICAL_EXP_LSB        
#     METRIC_WORST_SAMPLE_LIN_MSB = WORST_SAMPLE_LIN_MSB    
#     METRIC_WORST_SAMPLE_LIN_LSB = WORST_SAMPLE_LIN_LSB        
#     METRIC_WORST_SAMPLE_EXP_MSB = WORST_SAMPLE_EXP_MSB    
#     METRIC_WORST_SAMPLE_EXP_LSB = WORST_SAMPLE_EXP_LSB
#     METRIC_E4_MSB = E4_MSB                  
#     METRIC_E4_LSB = E4_LSB
#     METRIC_E5_MSB = E5_MSB                  
#     METRIC_E5_LSB = E5_LSB
#     METRIC_E6_MSB = E6_MSB                  
#     METRIC_E6_LSB = E6_LSB
#     METRIC_E7_MSB = E7_MSB                  
#     METRIC_E7_LSB = E7_LSB