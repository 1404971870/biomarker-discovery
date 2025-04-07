import glob
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class customDataset(Dataset):
    def __init__(self, root, transforms=None):
        # 保存两个不同的变换
        self.transform = transforms
        root = os.path.abspath(root)

        # 读取MSI文件夹中的图像文件路径
        self.files = sorted(glob.glob(os.path.join(root, "*.*")))

    def __getitem__(self, index):
        # 打开MSI中的图像
        img_a = Image.open(self.files[index]).convert('RGB')

        # 应用不同的转换
        img_a = self.transform(img_a)

        return {"A": img_a}


    def __len__(self):
        # 返回文件夹2中的图像数量
        return len(self.files)