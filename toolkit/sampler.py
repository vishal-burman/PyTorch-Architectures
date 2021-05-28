from torch.utils.data import Sampler

class SortishSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.batch_size, self.shuffle = data, batch_size, shuffle

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))
