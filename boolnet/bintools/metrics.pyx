# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

cpdef enum Metric:
    E1
    E2M, E2L,    # weighted
    E3M, E3L,    # hierarchical
    E4M, E4L,
    E5M, E5L,
    E6M, E6L,
    E7M, E7L,    # worst example
    ACCURACY,
    PER_OUTPUT


def all_metrics():
    return [E1, E2M, E2L, E3M, E3L, E4M, E4L, E5M, E5L,
            E6M, E6L, E7M, E7L, ACCURACY, PER_OUTPUT]


def all_metric_names():
    return [metric_name(m) for m in all_metrics()]


def metric_from_name(name):
    if name == 'e1':            return E1
    elif name == 'e2M':         return E2M
    elif name == 'e2L':         return E2L
    elif name == 'e3M':         return E3M
    elif name == 'e3L':         return E3L
    elif name == 'e4M':         return E4M
    elif name == 'e4L':         return E4L
    elif name == 'e5M':         return E5M
    elif name == 'e5L':         return E5L
    elif name == 'e6M':         return E6M
    elif name == 'e6L':         return E6L
    elif name == 'e7M':         return E7M
    elif name == 'e7L':         return E7L
    elif name == 'accuracy':    return ACCURACY
    elif name == 'per_output':  return PER_OUTPUT


def metric_name(metric):
    if metric == E1:            return 'e1'
    elif metric == E2M:         return 'e2M' 
    elif metric == E2L:         return 'e2L'
    elif metric == E3M:         return 'e3M' 
    elif metric == E3L:         return 'e3L'
    elif metric == E4M:         return 'e4M' 
    elif metric == E4L:         return 'e4L'
    elif metric == E5M:         return 'e5M' 
    elif metric == E5L:         return 'e5L'
    elif metric == E6M:         return 'e6M' 
    elif metric == E6L:         return 'e6L'
    elif metric == E7M:         return 'e7M' 
    elif metric == E7L:         return 'e7L'
    elif metric == ACCURACY:    return 'accuracy'
    elif metric == PER_OUTPUT:  return 'per_output'
