from BoolNet.Packing import pack_bool_matrix


class BoolMapping:
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

    def toDict(self):
        return {'inputs': self.inputs, 'target': self.target, 'Ne': self.Ne}