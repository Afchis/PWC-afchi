import os
import numpy as np

from PIL import Image
import flowiz as fz

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SmallData(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "dataloader/data/"# 
        self.names_flow = sorted(os.listdir(self.data_path))[0::3]
        self.names_img1 = sorted(os.listdir(self.data_path))[1::3]
        self.names_img2 = sorted(os.listdir(self.data_path))[2::3]
        self.img2tensor = transforms.ToTensor()
        
    def __len__(self):
        return 6

    def __getitem__(self, idx):
        img1 = Image.open(self.data_path + self.names_img1[idx])
        img2 = Image.open(self.data_path + self.names_img2[idx])
        img1 = self.img2tensor(img1) 
        img2 = self.img2tensor(img2)
        flow = torch.from_numpy(fz.read_flow(self.data_path + self.names_flow[idx])).permute(2, 0, 1)
        return img1, img2, flow


def Loader(batch_size, num_workers, shuffle=True):
    train_data = SmallData()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)
    return train_loader


# loader = Loader(batch_size=1, num_workers=2)
# for i, data in enumerate(loader):
#     img1, img2, flow = data
#     print(i, img1.shape, img2.shape, flow.shape)
if __name__ == "__main__":
    data = SmallData()
    img1, img2, flow = data[0]

