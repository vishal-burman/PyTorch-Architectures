from torch.utils.data import Sampler
import numpy as np


class SortishSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data, bs, shuffle=True):
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
    sort_idx = (
        np.concatenate(np.random.permutation(ck_idx[1:]))
        if len(ck_idx) > 1
        else np.array([], dtype=np.int)
    )
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx
