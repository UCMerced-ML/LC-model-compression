from .base_types import CompressionTypeBase
from .low_rank import LowRank


class Mapper(CompressionTypeBase):
    def __init__(self, list_of_compressions):
        """
        This class is stateful!
        """
        super(Mapper, self).__init__()
        assert all(map(lambda x: isinstance(x, CompressionTypeBase), list_of_compressions))

        self.list_of_compressions = list_of_compressions

    def load_state_dict(self, state_dict):
        self._state = state_dict
        for i, compression in enumerate(self.list_of_compressions):
            compression.load_state_dict(state_dict[f'mapped_compression_{i}'])

    def uncompress_state(self):
        return_list = []
        for i, compression in enumerate(self.list_of_compressions):
            return_list.append(compression.uncompress_state())
        return return_list

    def compress(self, data):
        if len(data) == len(self.list_of_compressions):
            # compression for every param
            return_list = []
            self.info[self.step_number] = []
            info = {}
            state = {}
            for i, (datum, compression) in enumerate(zip(data, self.list_of_compressions)):
                compression.step_number = self.step_number
                original_shape = datum.shape
                if isinstance(compression, LowRank):
                    realizable_mat = compression.compress(datum)
                    return_list.append(realizable_mat)
                else:
                    vector = datum.flatten()
                    realizable_vector = compression.compress(vector)
                    realizable_datum = realizable_vector.reshape(original_shape)
                    return_list.append(realizable_datum)
                info[f'mapped_compression_{i}'] = compression.info
                state[f'mapped_compression_{i}'] = compression.state_dict

            self.info[self.step_number].append(info)
            self._state = state

            return return_list
        else:
            raise Exception("# compressions should match # of datapoints")