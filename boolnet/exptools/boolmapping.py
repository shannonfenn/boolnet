from boolnet.bintools.packing import pack_bool_matrix
from boolnet.bintools.example_generator import packed_from_operator


class FileBoolMapping:
    def __init__(self, inputs, target, Ne):
        if inputs.shape[0] != target.shape[0] != Ne:
            raise ValueError('Input ({}), target ({}), Ne ({}) should match.'
                             .format(inputs.shape[0], target.shape[0], Ne))
        self.packed_inputs = pack_bool_matrix(inputs)
        self.packed_target = pack_bool_matrix(target)
        self.Ne = Ne

    @property
    def Ni(self):
        return self.inputs.shape[0]

    @property
    def No(self):
        return self.target.shape[0]


class OperatorBoolMapping:
    def __init__(self, indices, Nb, Ni, No, window_size, operator, N):
        if Nb > Ni:
            raise ValueError('Nb ({}) > Ni ({}).'.format(Nb, Ni))
        self.Nb = Nb
        self.Ni = Ni
        self.No = No
        self.window_size = window_size
        self.operator = operator
        self.N = N
        self._packed_repr = None

    @property
    def packed_inputs(self):
        if self.packed_repr is None:
            self.packed_repr = packed_from_operator(
                self.indices, self.Nb, self.No, self.operator, self.N)
        return self.packed_repr[0]

    @property
    def packed_target(self):
        if self.packed_repr is None:
            self.packed_repr = packed_from_operator(
                self.indices, self.Nb, self.No, self.operator, self.N)
        return self.packed_repr[1]

    @property
    def Ne(self):
        if self.packed_repr is None:
            self.packed_repr = packed_from_operator(
                self.indices, self.Nb, self.No, self.operator, self.N)
        return self.packed_repr[2]
