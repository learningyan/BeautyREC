import pdb
from .dataset import MTDataset
# from .wilddataset import WildDataset
import torch
traindataset = MTDataset('train')
traindataloader = torch.utils.data.DataLoader(
            traindataset,
            batch_size = 1,
            num_workers = 4)


testdataset = MTDataset('test')
testdataloader = torch.utils.data.DataLoader(
            testdataset,
            batch_size = 1,
            num_workers = 4)
