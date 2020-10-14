import os
import glob
import random
import numpy as np

from PIL import Image
import flowiz as fz

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FlyingChairs_train(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "dataloader/data/FlyingChairs/FlyingChairs_release/data/"
        self.flow_names = sorted(os.listdir(self.data_path))[::3]
        self.img1_names = sorted(os.listdir(self.data_path))[1::3]
        self.img2_names = sorted(os.listdir(self.data_path))[2::3]
        self.img2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.flow_names)

    def _random_crop(self, img1, img2, flow, h, w):
        _, h_old, w_old = img1.size()
        H, W = random.randint(0, h_old-h), random.randint(0, w_old-w)
        return img1[:, H:H+h, W:W+w], img2[:, H:H+h, W:W+w], flow[:, H:H+h, W:W+w]

    def _center_crop(self, img1, img2, flow, h, w):
        _, h_old, w_old = img1.size()
        H, W = int((h_old-h)/2), int((w_old-w)/2)
        return img1[:, H:H+h, W:W+w], img2[:, H:H+h, W:W+w], flow[:, H:H+h, W:W+w]

    def __getitem__(self, idx):
        img1 = Image.open(self.data_path + self.img1_names[idx])
        img2 = Image.open(self.data_path + self.img2_names[idx])
        img1, img2 = self.img2tensor(img1), self.img2tensor(img2)
        flow = torch.from_numpy(fz.read_flow(self.data_path + self.flow_names[idx])).permute(2, 0, 1)
        img1, img2, flow = self._random_crop(img1, img2, flow, h=384, w=448)
        return img1, img2, flow


class FlyingChairs_valid(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "dataloader/data/FlyingChairs/FlyingChairs2/val/"
        self.flow_names = sorted(os.listdir(self.data_path))[::14]
        self.img1_names = sorted(os.listdir(self.data_path))[2::14]
        self.img2_names = sorted(os.listdir(self.data_path))[3::14]
        self.img2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.flow_names)

    def _random_crop(self, img1, img2, flow, h, w):
        _, h_old, w_old = img1.size()
        H, W = random.randint(0, h_old-h), random.randint(0, w_old-w)
        return img1[:, H:H+h, W:W+w], img2[:, H:H+h, W:W+w], flow[:, H:H+h, W:W+w]

    def _center_crop(self, img1, img2, flow, h, w):
        _, h_old, w_old = img1.size()
        H, W = int((h_old-h)/2), int((w_old-w)/2)
        return img1[:, H:H+h, W:W+w], img2[:, H:H+h, W:W+w], flow[:, H:H+h, W:W+w]

    def __getitem__(self, idx):
        img1 = Image.open(self.data_path + self.img1_names[idx])
        img2 = Image.open(self.data_path + self.img2_names[idx])
        img1, img2 = self.img2tensor(img1), self.img2tensor(img2)
        flow = torch.from_numpy(fz.read_flow(self.data_path + self.flow_names[idx])).permute(2, 0, 1)
        img1, img2, flow = self._random_crop(img1, img2, flow, h=384, w=448)
        return img1, img2, flow


class MpiSintel(Dataset):
    def __init__(self):
        super().__init__()
        self.flow_names = sorted(glob.glob(os.path.join("dataloader/data/MpiSintel/training/flow/", "*/*.flo")))
        self.img2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.flow_names)

    def _random_crop(self, img1, img2, flow, h, w):
        _, h_old, w_old = img1.size()
        H, W = random.randint(0, h_old-h), random.randint(0, w_old-w)
        return img1[:, H:H+h, W:W+w], img2[:, H:H+h, W:W+w], flow[:, H:H+h, W:W+w]

    def _center_crop(self, img1, img2, flow, h, w):
        _, h_old, w_old = img1.size()
        H, W = int((h_old-h)/2), int((w_old-w)/2)
        return img1[:, H:H+h, W:W+w], img2[:, H:H+h, W:W+w], flow[:, H:H+h, W:W+w]

    def __getitem__(self, idx):
        img1 = Image.open(self.flow_names[idx][:35] + "clean" + self.flow_names[idx][39:-4] + ".png")
        img2_number = "%04d"%(int(self.flow_names[idx][-8:-4])+1)
        img2 = Image.open(self.flow_names[idx][:35] + "clean" + self.flow_names[idx][39:-8] + img2_number + ".png")
        img1 = self.img2tensor(img1)
        img2 = self.img2tensor(img2)
        flow = torch.from_numpy(fz.read_flow(self.flow_names[idx])).permute(2, 0, 1)
        img1, img2, flow = self._center_crop(img1, img2, flow, h=384, w=768)
        return img1, img2, flow


def Loader(dataset, batch_size, num_workers, shuffle=True):
    if dataset == "FlyingChairs":
        train_data = FlyingChairs_train()
        valid_data = FlyingChairs_valid()
    elif dataset == "MpiSintel":
        train_data = MpiSintel()
        valid_data = FlyingChairs_valid()
    else:
        print("Error: Wrong dataset name")
        quit()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=shuffle)
    valid_loader = DataLoader(dataset=valid_data,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=False)
    return train_loader, valid_loader


if __name__ == "__main__":
    # data = MpiSintel()
    # img1, img2, flow = data[0]
    # print(img1.shape, img2.shape, flow.shape)
    data = FlyingChairs_train()
    print(len(data))
    data[0]

