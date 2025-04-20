from pathlib import Path
from itertools import chain
import os

from munch import Munch
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms





import random

def listdir(dname):
    extensions = ['png', 'jpg', 'jpeg', 'JPG']
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) 
                                for ext in extensions]))
    return fnames               # PATH 객체

class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        print("--------------------- Dataset Loaded ---------------------")
        domains = [d for d in os.listdir(root) if not d.startswith('.')]                      # duhyeonkim updated : hidden folder restricted | 재귀적이지 않고, 바로 아래 dir만 불러옴
        fnames, fnames2 = [], []
        for idx, domain in enumerate(sorted(domains, reverse=True)):             # duhyeonkim updated : for my own dataset, not named as exp and raw | 250227 reverse=True deleted for exp->raw GAN testing
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)             # 'raw' or 'exp' 안의 모든 이미지 파일 PATH 값
            if idx == 0:                                # exp
                fnames += cls_fnames
            elif idx == 1:                              # raw
                fnames2 += cls_fnames

            # duhyeonkim updated (leveraging random pair generation)
            random.shuffle(fnames)
            random.shuffle(fnames2)

        return list(zip(fnames, fnames2))               # zip 객체는 한 번 순회하면 다시 사용불가 -> 기본적으로 (idx, [] or tuple or scalar etc..)

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]             # self.samples = (exp, raw)
        name = str(fname2)
        img_name, _ = name.split('.', 1)                # 1번만 spilt한다.
        _, img_name = img_name.rsplit('/', 1)           # raw 파일명만 추출
        # img = Image.open(fname).convert('RGB')
        # img2 = Image.open(fname2).convert('RGB')
        # if self.transform is not None:
        #     img = self.transform(img)
        #     img2 = self.transform(img2)

        with Image.open(fname) as img, Image.open(fname2) as img2:
            img = img.convert('RGB')
            img2 = img2.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)

        return img, img2, img_name                      # exp, raw, raw image filename

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, img_size=512, resize_size=256, batch_size=8, shuffle=True, num_workers=8, drop_last=True):

    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.Resize([resize_size, resize_size]),
        # transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)         # get_item : exp, raw, raw image filename

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                        #    pin_memory=True,           # mps device
                           drop_last=drop_last)


def get_test_loader(root, img_size=512, batch_size=8, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.CenterCrop([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                        #    pin_memory=True
                           )


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        # duhyeonkim updated : loader intitialize here
        self.iter = iter(self.loader)

    def _fetch_refs(self):
        try:
            x, y, name = next(self.iter)
        except (AttributeError, StopIteration):
            print("------------------- !! new iter -------------------")
            self.iter = iter(self.loader)
            x, y, name = next(self.iter)            # iterator 만들고 첫번째 batch
        return x, y, name

    def __next__(self):
        x, y, img_name = self._fetch_refs()
        x, y = x.to(self.device), y.to(self.device)
        inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)     # get_item : exp, raw, raw image filename
        
        return inputs

