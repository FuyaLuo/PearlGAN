import torch.utils.data
from data.EBGC_dataset import EBGCDataset


class DataLoader():
    def name(self):
        return 'DataLoader'

    def __init__(self, opt):
        self.opt = opt
        self.dataset = EBGCDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

