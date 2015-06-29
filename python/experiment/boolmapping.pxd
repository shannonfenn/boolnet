cdef inline size_t bin_add(size_t index, size_t bit_width):
    return (index // bit_width) + (index % bit_width))

cdef inline size_t bin_sub(size_t index, size_t bit_width):
    return (index // bit_width) - (index % bit_width))

cdef inline size_t bin_mul(size_t index, size_t bit_width):
    return (index // bit_width) * (index % bit_width))

cdef inline size_t bin_div(size_t index, size_t bit_width):
    return (index // bit_width) // (index % bit_width))

cdef inline size_t bin_mod(size_t index, size_t bit_width):
    return (index // bit_width) % (index % bit_width))

cdef class BoolMapping:
    def __init__(self, inputs, target, Ne)
    @staticmethod
    def _validate(inputs, target, Ne):
        if inputs.shape[0] != target.shape[0] != Ne:
            raise ValueError(
                'Dataset input (), target () and Ne () do not match.'.format(
                    inputs.shape[0], target.shape[0], Ne))
        return pack_bool_matrix(inputs), pack_bool_matrix(target)

    @property
    def Ni(self):
        return self.inputs.shape[0]

    @property
    def No(self):
        return self.target.shape[0]

    def toDict(self)