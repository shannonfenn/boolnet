from boolnet.bintools.packing import pack_bool_matrix
from collections import namedtuple


class FileBoolMapping:
    def __init__(self, inputs, target, Ne):
        self.inputs, self.target = self._validate(inputs, target, Ne)
        self.Ne = Ne

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


OperatorBoolMapping = namedtuple(
    'OperatorBoolMapping', [
        'indices',
        'Nb',
        'Ni',
        'No',
        'window_size',
        'operator',
        'N'])
