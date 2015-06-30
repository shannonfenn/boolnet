from BoolNet.BitError import (E1, E2L, E2M, E3L, E3M, E4L, E4M, E5L, E5M,
                              E6L, E6M, E7L, E7M, ACCURACY, PER_OUTPUT, Metric)
import numpy as np
cimport numpy as np
import cython
from BoolNet.PopCount cimport popcount_matrix, popcount_vector
from BoolNet.Packing cimport packed_type_t, PACKED_SIZE, PACKED_ALL_SET
from BoolNet.Packing import packed_type


chained_evaluators = {
    E1: ChainedEvaluator,
    E2L: ChainedE2
    E2M: ChainedE2
    E3L: ChainedE3
    E3M: ChainedE3
    E4L: ChainedE4
    E4M: ChainedE4
    E5L: ChainedE5
    E5M: ChainedE5
    E6L: ChainedE6
    E6M: ChainedE6
    E7L: ChainedE7
    E7M: ChainedE7
    ACCURACY: ChainedAccuracu
    PER_OUTPUT: ChainedPerOutput
}


cdef class ChainedEvaluator():
    cdef:
        np.uint64_t[:] row_accumulator
        size_t Ne, No, cols, start, step

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        self.Ne = Ne
        self.No = No
        self.cols = cols
        if msb:
            self.start, self.step = No-1, -1
        else:
            self.start, self.step = 0, 1
        self.row_accumulator = np.zeros(No, dtype=np.uint64)

    cpdef void reset(self):
        row_accumulator[:] = 0

    cpdef void partial_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        for i in range(No):
            self.row_accumulator[i] += popcount_vector(E[i, :])

    cpdef final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E)

        for i in range(No):
            result += self.row_accumulator[i] / <double>self.Ne

        return result / No


cdef class ChainedE2(ChainedEvaluator):
    cdef:
        double[:] weight_vector

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)
        if msb:
            self.weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            self.weight_vector = np.arange(No, 0, -1, dtype=np.float64)

    def final_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i
        cdef double result = 0.0

        self.partial_evaluation(E)
            
        for i in range(No):
            result += self.row_accumulator[i] * self.weight_vector[i] / <double>self.Ne

        return result * 2.0 / (No + 1.0) / No



cdef class ChainedE3(ChainedEvaluator):

    def __init__(self, size_t Ne, size_t No, size_t cols, bint msb):
        super().__init__(Ne, No, cols, msb)

    def partial_evaluation(self, packed_type_t[:, ::1] E):
        cdef size_t i, r
        self.row_accumulator[self.start] += popcount_vector(E[self.start, :])
        r = start + step
        for i in range(No-1):
            self.row_accumulator[i] += popcount_vector(E[r, :] | E[r-step, :])


cdef class ChainedE4(ChainedEvaluator):
    cdef:
        size_t end_mask_val
        packed_type_t[:] masks

    def __init__(self, size_t Ne, size_t No, size_t cols,
                 bint msb, packed_type_t end_mask):
        super().__init__(Ne, No, cols, msb)
        self.masks = np.zeros(No, dtype=packed_type)
        self.end_mask = end_mask

    def partial_evaluation(self, packed_type_t[:, ::1] E):
        cdef:
            size_t i, r, c, err_col, prev_err_col
            packed_type_t mask = 0

        prev_err_col = cols # first start with no indicated error
        
        r = self.start
        for i in range(No):
            if self.row_accumulator[r] >= 0:
                self.row_accumulator[r] += PACKED_SIZE * cols 
            else:
                # find first example with non zero error 
                err_col = prev_err_col
                for c in range(prev_err_col):
                    if E[r, c]:
                        err_col = c
                    break
                # if we aren't past the end of the row
                if err_col != cols:
                    # mask needs recalculating if we have found an earlier
                    # error column or an earlier example in this column
                    if err_col != prev_err_col or mask < E[r, c]:
                        mask = cascade(E[r, c])
                    # treat all subsequent examples as if high
                    self.row_accumulator[r] += popcount_chunk(mask) + PACKED_SIZE * (cols - err_col - 1)
                    prev_err_col = err_col
            r = self.start + self.step


    def final_evaluation(self, packed_type_t[:, ::1] E):
        cdef:
            size_t i, r
            double result = 0.0
            packed_type_t mask = 0

        self.partial_evaluation(E[:, -1:])

        r = self.start
        for i in range(No):
            if self.row_accumulator[r] >= 0:
                self.row_accumulator[r] += end_mask_val
            else:
                mask |= E[r, -1] | end_mask
                self.row_accumulator[r] += popcount_chunk(mask)
            r = self.start + self.step
            result += self.row_accumulator[r] / <double>self.Ne
        
        return result / No


cdef class ChainedE5(ChainedEvaluator):
    cdef:
        size_t end_mask_val
        packed_type_t[:] masks

    def __init__(self, size_t Ne, size_t No, size_t cols,
                 bint msb, packed_type_t end_mask):
        super().__init__(Ne, No, cols, msb)
        self.masks = np.zeros(No, dtype=packed_type)
        self.end_mask = end_mask

    def partial_evaluation(self, packed_type_t[:, ::1] E):
        cdef:
            size_t i, r, c
            bint zero_last, found = False

        prev_err_col = cols # first start with no indicated error
        
        r = self.start
        for i in range(No):
            found ||= self.row_accumulator[r] >= 0
            if found:
                self.row_accumulator[r] += PACKED_SIZE * cols 
            else:
                bits = flood_count_vector(E[r, :])
                self.row_accumulator[r] += bits
                found = bits != 0
            r = self.start + self.step


    def final_evaluation(self, packed_type_t[:, ::1] E):
        cdef:
            size_t i, r
            double result = 0.0
            packed_type_t mask = 0

        self.partial_evaluation(E[:, -1:])

        r = self.start
        for i in range(No):
            if self.row_accumulator[r] >= 0:
                self.row_accumulator[r] += end_mask_val
            else:
                mask |= E[r, -1] | end_mask
                self.row_accumulator[r] += popcount_chunk(mask)
            r = self.start + self.step
            result += self.row_accumulator[r] / <double>self.Ne
        
        return result / No


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline packed_type_t cascade(packed_type_t word):
    word = word | word - 1
    word |= word >> 1
    word |= word >> 2
    word |= word >> 4
    word |= word >> 8
    word |= word >> 16
    word |= word >> 32
    return word


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void E3(packed_type_t[:, ::1] E_in,
             packed_type_t[:, ::1] E_out,
             bint msb):
    cdef:
        size_t rows, cols, c, i, r, start
        int step
        
    rows, cols = E_in.shape[0], E_in.shape[1]

    if msb:
        start, step = rows-1, -1
    else:
        start, step = 0, 1
    
    # first feature is simply copied
    for c in range(cols):
        E_out[start, c] = E_in[start, c]

    # subsequent features are bitwise ORed with the previous
    r = start + step
    for i in range(rows-1):
        for c in range(cols):
            E_out[r, c] = E_out[r-step, c] | E_in[r, c]
        r += step


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void E4(packed_type_t[:, ::1] E_in,
             packed_type_t[:, ::1] E_out,
             bint[:] cin, bint[:] cout, bint msb):
    cdef:
        size_t rows, cols, start, c, r, i, prev_err_col, err_col
        int step
        packed_type_t mask
        
    rows, cols = E_in.shape[0], E_in.shape[1]

    if msb:
        start, step = rows-1, -1
    else:
        start, step = 0, 1
    
    # first start with no indicated error
    prev_err_col = cols
    # mask initially unset
    mask = 0

    r = start
    for i in range(rows):
        if cin[r]:
            # for this feature there is a carry in error
            # set all examples high and all examples for
            # future features high
            mask = PACKED_ALL_SET
            err_col = 0
            E_out[r, :cols] = PACKED_ALL_SET
        else:
            # find first example with non zero error 
            err_col = prev_err_col
            for c in range(prev_err_col):
                E_out[r, c] = E_in[r, c]
                if E_in[r, c]:
                    err_col = c
                    break
            # mask needs recalculating if we have found an earlier
            # error column or an earlier example in this column
            if err_col != prev_err_col or mask < E_in[r, c]:
                mask = cascade(E_in[r, c])

            # if we aren't at the end of the row
            if err_col != cols:
                E_out[r, err_col] = mask
                # set all subsequent examples high
                E_out[r, err_col+1:cols] = PACKED_ALL_SET

        prev_err_col = err_col
        # calculate error carry out
        cout[r] = E_out[r, -1]
        r += step


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void E5(packed_type_t[:, ::1] E_in,
             packed_type_t[:, ::1] E_out,
             bint[:] cin, bint[:] cout, bint msb):
    cdef:
        unsigned int rows, cols, start, c, r, i, err_col
        int step
        bint found = False
        
    rows, cols = E_in.shape[0], E_in.shape[1]

    if msb:
        start, step = rows-1, -1
    else:
        start, step = 0, 1
    
    err_col = cols

    r = start
    for i in range(rows):
        if cin[r]:
            # for this feature there is a carry in error
            # set all examples high and all examples for
            # future features high
            mask = PACKED_ALL_SET
            err_col = 0
            E_out[r, :cols] = PACKED_ALL_SET
        else:
            # find first example with non zero error 
            for c in range(cols):
                E_out[r, c] = E_in[r, c]
                found = <bint>E_in[r, c]
                if found:
                    err_col = c
                    break
            if found:
                # need to cascade 1s in this column
                E_out[r, err_col] = cascade(E_in[r, c])
                # set all subsequent examples high
                E_out[r, err_col+1:cols] = PACKED_ALL_SET
                break
        r += step

    # set all subsequent features high
    if found:
        rows = r if msb else rows - r - 1
        r += step
        for i in range(rows):
            E_out[r, :cols] = PACKED_ALL_SET
            r += step

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void E6(packed_type_t[:, ::1] E_in,
             packed_type_t[:, ::1] E_out,
             bint msb):
    cdef:
        unsigned int rows, cols, c, i, r, start
        int step
        bint found = False
        
    rows, cols = E_in.shape[0], E_in.shape[1]

    if msb:
        start, step = rows-1, -1
    else:
        start, step = 0, 1
    
    r = start
    for i in range(rows):
        # find first feature with non zero error 
        for c in range(cols):
            E_out[r, c] = E_in[r, c]
            found = found or <bint>E_in[r, c]
        if found:
            break
        r += step

    if found:
        # set all subsequent features high
        rows = r if msb else rows - r - 1
        r += step
        for i in range(rows):
            E_out[r, :cols] = PACKED_ALL_SET
            r += step


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void E7(packed_type_t[:, ::1] E_in,
             packed_type_t[:, ::1] E_out,
             bint msb):
    cdef:
        unsigned int rows, cols, c, i, r, start
        int step
        bint found = False
        
    rows, cols = E_in.shape[0], E_in.shape[1]

    if msb:
        start, step = rows-1, -1
    else:
        start, step = 0, 1
    
    r = start

    for i in range(rows):
        # find first feature with non zero error 
        for c in range(cols):
            found = <bint>E_in[r, c]
            if found:
                break
            E_out[r, c] = 0
        if found:
            break
        r += step

    if found:
        # set this and all subsequent features high
        rows = r + 1 if msb else rows - r
        for i in range(rows):
            E_out[r, :cols] = PACKED_ALL_SET
            r += step


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double accuracy(packed_type_t[:, ::1] E_in,
                     packed_type_t[:, ::1] E_out,
                     size_t Ne):
    cdef:
        size_t rows, cols, r, c
    
    rows, cols = E_in.shape[0], E_in.shape[1]

    # this uses the first row of output matrix as an accumulator
    # so initialise that row
    E_out[0, :] = 0

    # accumulate bitwise or of all rows
    for r in range(rows):
        for c in range(cols):
            E_out[0, c] |= E_in[r, c]

    # the accuracy is the 1 minus the mean of the accumulator
    return 1.0 - (popcount_vector(E_out[0, :]) / <double>Ne)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void apply_end_mask(packed_type_t[:, ::1] E, packed_type_t end_mask):
    cdef:
        size_t rows, c, r

    rows = E.shape[0]
    c = E.shape[1] - 1

    for r in range(rows):
        E[r, c] &= end_mask




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def partial_metric_value(packed_type_t[:, ::1] E_in,
                         packed_type_t[:, ::1] E_out,
                         unsigned int Ne,
                         packed_type_t end_mask,
                         metric):
    cdef:
        unsigned int o, No
        double[:] weight_vector, totals

    No = E_in.shape[0]
    # ################# SPECIAL METRICS ################# #

    if metric == PER_OUTPUT:
        totals = np.zeros(No, dtype=np.float64)
        for o in range(No):
            totals[o] = popcount_vector(E_in[o, :]) / <double>Ne
        return totals

    elif metric == ACCURACY:
        return accuracy(E_in, E_out, Ne)

    elif metric == E2M or metric == E2L:
        if metric == E2M:
            weight_vector = np.arange(1, No+1, dtype=np.float64)
        else:
            weight_vector = np.arange(No, 0, -1, dtype=np.float64)
        
        return dot_mean(E_in, weight_vector, Ne) * 2.0 / (No + 1.0)

    # ################ STANDARD METRICS ################ #

    elif metric == E1:
        return popcount_matrix(E_in) / <double>No / Ne

    elif metric == E3M or metric == E3L:
        E3(E_in, E_out, metric==E3M)

    elif metric == E4M or metric == E4L:
        E4(E_in, E_out, metric==E4M)

    elif metric == E5M or metric == E5L:
        E5(E_in, E_out, metric==E5M)

    elif metric == E6M or metric == E6L:
        E6(E_in, E_out, metric==E6M)

    elif metric == E7M or metric == E7L:
        E7(E_in, E_out, metric==E7M)

    else:
        raise ValueError('Invalid metric - {}'.format(metric))

    # Same for E3, 4, 5, 6, 7
    apply_end_mask(E_out, end_mask)
    return popcount_matrix(E_out) / <double>No / Ne
