# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

def all_functions():
    return scalar_functions() + per_output_functions()


def scalar_functions():
    return [E1, E2M, E2L, E3M, E3L, E4M, E4L, E5M, E5L,
            E6M, E6L, E7M, E7L, ACCURACY, MCC]


def per_output_functions():
    return [PER_OUTPUT_ERROR, PER_OUTPUT_MCC]


def all_function_names():
    return scalar_function_names() + per_output_function_names()


def scalar_function_names():
    return [function_name(m) for m in scalar_functions()]


def per_output_function_names():
    return [function_name(m) for m in per_output_functions()]


def function_from_name(name):
    if name == 'e1':                    return E1
    elif name == 'e2M':                 return E2M
    elif name == 'e2L':                 return E2L
    elif name == 'e3M':                 return E3M
    elif name == 'e3L':                 return E3L
    elif name == 'e4M':                 return E4M
    elif name == 'e4L':                 return E4L
    elif name == 'e5M':                 return E5M
    elif name == 'e5L':                 return E5L
    elif name == 'e6M':                 return E6M
    elif name == 'e6L':                 return E6L
    elif name == 'e7M':                 return E7M
    elif name == 'e7L':                 return E7L
    elif name == 'accuracy':            return ACCURACY
    elif name == 'mcc':                 return MCC
    elif name == 'per_output_error':    return PER_OUTPUT_ERROR
    elif name == 'per_output_mcc':      return PER_OUTPUT_MCC
    else: raise ValueError('No function named \'{}\''.format(name))


def function_name(function_id):
    if function_id == E1:                   return 'e1'
    elif function_id == E2M:                return 'e2M' 
    elif function_id == E2L:                return 'e2L'
    elif function_id == E3M:                return 'e3M' 
    elif function_id == E3L:                return 'e3L'
    elif function_id == E4M:                return 'e4M' 
    elif function_id == E4L:                return 'e4L'
    elif function_id == E5M:                return 'e5M' 
    elif function_id == E5L:                return 'e5L'
    elif function_id == E6M:                return 'e6M' 
    elif function_id == E6L:                return 'e6L'
    elif function_id == E7M:                return 'e7M' 
    elif function_id == E7L:                return 'e7L'
    elif function_id == ACCURACY:           return 'accuracy'
    elif function_id == MCC:                return 'mcc'
    elif function_id == PER_OUTPUT_ERROR:   return 'per_output_error'
    elif function_id == PER_OUTPUT_MCC:     return 'per_output_mcc'
    else: raise ValueError('No function id \'{}\''.format(function_id))
