import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from src.utils.preprocess_utils.make_samples import make_windows


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]


        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape
        stride = slide_stride if is_train else 1

        # shared windowing (single source of truth across both pipelines);
        # make_windows expects (T, N) and yields (num, N, w) / (num, N).
        xs, ys = make_windows(data.t().numpy(), slide_win, stride)
        idx = list(range(slide_win, total_time_len, stride))

        x = torch.from_numpy(xs).double().contiguous()
        y = torch.from_numpy(ys).double().contiguous()
        labels = torch.Tensor([float(labels[i]) for i in idx]).contiguous()

        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





