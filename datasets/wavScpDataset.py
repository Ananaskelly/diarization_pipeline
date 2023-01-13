import os
import kaldiio
from torch.utils.data import Dataset


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class WavScpDataset(Dataset):

    def __init__(self, scp_path):
        super(WavScpDataset, self).__init__()
        self._scp_dict = kaldiio.load_scp(scp_path)
        self._scp_keys = list(self._scp_dict.keys())

    def __getitem__(self, item):
        utt_id = self._scp_keys[item]
        rate, data = self._scp_dict[utt_id]

        return data


dc = WavScpDataset('../meta_files/voxceleb_meta/wav.scp')

for val in dc:
    print(val)
    exit(0)
