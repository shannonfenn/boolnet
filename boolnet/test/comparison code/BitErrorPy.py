import numpy as np
from enum import Enum, unique


@unique
class Function(Enum):
    E1 = 1,         # simple
    E2M = 2,     # weighted
    E2L = 3,
    E3M = 4,     # hierarchical
    E3L = 5,
    E4M = 8,
    E4L = 9,
    E5M = 10,
    E5L = 11,
    E6M = 12,
    E6L = 13,
    E7M = 6,     # worst example
    E7L = 7,
    ACCURACY = 14,
    PER_OUTPUT = 15,

    def __str__(self):
        s = self.name
        if s.startswith('E') and s.endswith(('M', 'L')):
            return s[0].lower() + s[1:]
        else:
            return s.lower()

    def raw_str(self):
        return self.name

E1 = Function.E1
E2M = Function.E2M
E2L = Function.E2L
E3M = Function.E3M
E3L = Function.E3L
E4M = Function.E4M
E4L = Function.E4L
E5M = Function.E5M
E5L = Function.E5L
E6M = Function.E6M
E6L = Function.E6L
E7M = Function.E7M
E7L = Function.E7L
ACCURACY = Function.ACCURACY
PER_OUTPUT = Function.PER_OUTPUT


def all_functions():
    for m in Function:
        yield m


def all_function_names():
    for m in Function:
        yield str(m)


def function_from_name(name):
    return Function[name.upper()]


def function_name(metric):
    return str(metric)


def flood_count_msb(error_matrix, No):
    return ((No - np.argmax(np.fliplr(error_matrix), axis=1)) *
            (error_matrix.sum(axis=1) > 0))


def flood_count_lsb(error_matrix, No):
    return (No - np.argmax(error_matrix, axis=1)) * (error_matrix.sum(axis=1) > 0)


# Many of the calculations in this method rely on error_matrix
# only being comprised of 1s and 0s
def function_value(error_matrix, metric):
    error = 0.0
    # for weighted and hierarchical methods
    Ne, No = error_matrix.shape
    msb_vector = np.arange(No) + 1
    lsb_vector = np.flipud(msb_vector)
    weight_denominator = No*(No+1)/2

    # ################# MULTI-VALUED METRICS ################# #

    if metric == PER_OUTPUT:
        return np.mean(error_matrix, axis=0)

    # ############### SINGLE-VALUED METRICS ############### #

    if metric == E1:
        return error_matrix.mean()

    if metric == E2M:
        return np.dot(error_matrix, msb_vector).mean() / weight_denominator

    if metric == E2L:
        return np.dot(error_matrix, lsb_vector).mean() / weight_denominator

    if metric == ACCURACY:
        return 1.0 - np.mean(np.any(error_matrix, axis=1))

    if metric == E3M:
        return flood_count_msb(error_matrix, No).mean() / No

    if metric == E3L:
        return flood_count_lsb(error_matrix, No).mean() / No

    if metric == E4M or metric == E4L:
        # Find the msb/lsb in error for each sample
        if metric == E4M:
            errs = flood_count_msb(error_matrix, No)
        else:
            errs = flood_count_lsb(error_matrix, No)
        # make errors monotonic increasing
        for i in range(1, len(errs)):
            errs[i] = max(errs[i], errs[i-1])
        return sum(errs) / len(errs) / No

    if metric == E5M or metric == E5L:
        # Find the msb/lsb in error for each sample
        if metric == E5M:
            worst = flood_count_msb(error_matrix, No)
        else:
            worst = flood_count_lsb(error_matrix, No)
        # Find the first sample with the largest error
        worst_example = np.argmax(worst)
        largest_error = worst[worst_example]
        error = ((largest_error - 1) * worst_example +
                 largest_error * (Ne - worst_example))
        return error / Ne / No

    if metric == E6M or metric == E6L:
        # Find the msb/lsb in error for each sample
        if metric == E6M:
            worst = flood_count_msb(error_matrix, No)
        else:
            worst = flood_count_lsb(error_matrix, No)
        largest_error = np.amax(worst)
        num_wrong_on_worst_bit = np.count_nonzero(worst == largest_error)
        return (largest_error - 1 + num_wrong_on_worst_bit / Ne) / No

    if metric == E7M:
        return flood_count_msb(error_matrix, No).max() / No

    if metric == E7L:
        return flood_count_lsb(error_matrix, No).max() / No

    raise ValueError('Invalid metric - {}'.format(metric))
