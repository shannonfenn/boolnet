# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

def all_functions():
    return scalar_functions() + per_output_functions()


def scalar_functions():
    return [E1, E2, E3, E4, E5, E6, E7,
            ACCURACY, E1_MCC, E2_MCC, E6_MCC]


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
    elif name == 'e2':                  return E2
    elif name == 'e3':                  return E3
    elif name == 'e4':                  return E4
    elif name == 'e5':                  return E5
    elif name == 'e6':                  return E6
    elif name == 'e7':                  return E7
    elif name == 'accuracy':            return ACCURACY
    elif name == 'e1_mcc':              return E1_MCC
    elif name == 'e2_mcc':              return E2_MCC
    elif name == 'e6_mcc':              return E6_MCC
    elif name == 'per_output_error':    return PER_OUTPUT_ERROR
    elif name == 'per_output_mcc':      return PER_OUTPUT_MCC
    else: raise ValueError('No function named \'{}\''.format(name))


def function_name(function_id):
    if function_id == E1:                   return 'e1'
    elif function_id == E2:                 return 'e2'
    elif function_id == E3:                 return 'e3'
    elif function_id == E4:                 return 'e4'
    elif function_id == E5:                 return 'e5'
    elif function_id == E6:                 return 'e6'
    elif function_id == E7:                 return 'e7'
    elif function_id == ACCURACY:           return 'accuracy'
    elif function_id == E1_MCC:             return 'e1_mcc'
    elif function_id == E2_MCC:             return 'e2_mcc'
    elif function_id == E6_MCC:             return 'e6_mcc'
    elif function_id == PER_OUTPUT_ERROR:   return 'per_output_error'
    elif function_id == PER_OUTPUT_MCC:     return 'per_output_mcc'
    else: raise ValueError('No function id \'{}\''.format(function_id))


def is_minimiser(function_id):
    if function_id in [E1, E2, E3, E4, E5, E6, E7, PER_OUTPUT_ERROR]:
        return True
    elif function_id in [ACCURACY, E1_MCC, E2_MCC, E6_MCC, PER_OUTPUT_MCC]:
        return False
    else:
        raise ValueError('No function id \'{}\''.format(function_id))


def optimum(function_id):
    if function_id in [E1, E2, E3, E4, E5, E6, E7, PER_OUTPUT_ERROR]:
        return 0.0
    elif function_id in [ACCURACY, E1_MCC, E2_MCC, E6_MCC, PER_OUTPUT_MCC]:
        return 1.0
    else:
        raise ValueError('No function id \'{}\''.format(function_id))
