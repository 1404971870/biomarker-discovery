import glob
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 对于数据集中的每一张图像，将其水平分割成两半，形成两个图像：img_A（左半部分）和img_B（右半部分）
# 以50%的概率对img_A和img_B进行水平翻转
#
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

class customDataset(Dataset):
    def __init__(self, root_msi, root_low_wsi, mode, transforms_msi=None, transforms_wsi=None):
        # 保存两个不同的变换
        self.transform_msi = transforms.Compose(transforms_msi)
        self.transform_wsi = transforms.Compose(transforms_wsi)
        # self.transform_a = transforms.Compose(transforms_a)
        # self.transform_b = transforms.Compose(transforms_b)
        # 根据模式确定文件夹路径
        root_a = os.path.join(root_msi,  'normal_constrast')
        # root_b = os.path.join(root_high_wsi,  'wsi_high')
        root_b = os.path.join(root_low_wsi,  'wsi')

        # 读取MSI文件夹中的图像文件路径
        self.files_a = sorted(glob.glob(os.path.join(root_a, "*.*")))

        # 读取WSI文件夹中的图像文件路径
        self.files_b = sorted(glob.glob(os.path.join(root_b, "*.*")))

        # self.files_c = sorted(glob.glob(os.path.join(root_c, "*.*")))


    def __getitem__(self, index):
        # 打开MSI中的图像
        img_a = Image.open(self.files_a[index]).convert('RGB')
        img_a = img_a.rotate(-90, expand=True)
        # 打开WSI中的图像
        # 因为wsi每一张都是一样的，为了不复制8474张，所以把index去掉，只取第一张就完事了
        img_b = Image.open(self.files_b[0]).convert('RGB')
        # img_c = Image.open(self.files_c[0]).convert('RGB')

        # image_array = np.array(img_a)
        # # 将白色背景（灰度值为255）转换为黑色背景（灰度值为0）
        # image_array[image_array == 255] = 0
        # img_a = Image.fromarray(image_array)
        # wsi现在是正的
        # img_a = img_a.rotate(-90, expand=True)
        # 已经是534，534了
        # img_a = img_a.resize((534, 534), Image.BICUBIC)

        # 应用不同的转换
        img_a = self.transform_msi(img_a)
        img_b = self.transform_msi(img_b)
        # 因为要resize成256
        # img_c = self.transform_msi(img_c)

        return {"A": img_a, "B": img_b, "filename": os.path.splitext(os.path.basename(self.files_a[index]))[0]}


    def __len__(self):
        # 返回文件夹2中的图像数量
        return len(self.files_a)

class high_customDataset(Dataset):
    def __init__(self, root_msi,mode,angle, transforms_msi=None, transforms_wsi=None):
        # 保存两个不同的变换
        self.transform_msi = transforms.Compose(transforms_msi)
        # self.transform_a = transforms.Compose(transforms_a)
        # self.transform_b = transforms.Compose(transforms_b)
        # 根据模式确定文件夹路径
        root_a = root_msi
        self.angle = angle

        # 读取MSI文件夹中的图像文件路径
        self.files_a = sorted(glob.glob(os.path.join(root_a, "*.*")))

    def __getitem__(self, index):
        # 打开MSI中的图像
        img_a = Image.open(self.files_a[index]).convert('RGB')
        img_a = img_a.rotate(self.angle, expand=True)
        # 打开WSI中的图像
        # 因为wsi每一张都是一样的，为了不复制8474张，所以把index去掉，只取第一张就完事了

        # image_array = np.array(img_a)
        # # 将白色背景（灰度值为255）转换为黑色背景（灰度值为0）
        # image_array[image_array == 255] = 0
        # img_a = Image.fromarray(image_array)
        # wsi现在是正的
        # img_a = img_a.rotate(-90, expand=True)
        # 已经是534，534了
        # img_a = img_a.resize((534, 534), Image.BICUBIC)

        # 应用不同的转换
        img_a = self.transform_msi(img_a)
        # 因为要resize成256

        return {"A": img_a, "filename": os.path.splitext(os.path.basename(self.files_a[index]))[0]}


    def __len__(self):
        # 返回文件夹2中的图像数量
        return len(self.files_a)


class MSI_dataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.transform = self._define_transforms()
        self.image_names = []  # 添加一个列表来存储图像的文件名

    def _define_transforms(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小为256x256
            # transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
            transforms.ToTensor(),           # 将PIL图像转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])

    def load_images(self):
        images = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith((".jpg", ".png")):  # 检查文件扩展名，忽略大小写
                file_path = os.path.join(self.folder_path, filename)
                self.image_names.append(filename)  # 添加文件名到列表
                image = Image.open(file_path)
                image = image.rotate(-90, expand=True)
                image_tensor = self.transform(image)
                images.append(image_tensor)
        return images

    def create_image_tensor(self):
        images = self.load_images()
        images_tensor = torch.cat(images, dim=0)  # 在通道维度进行叠加
        return images_tensor.unsqueeze(0)  # 添加批次通道

    def get_image_names(self):
        self.load_images()  # 确保图像和文件名列表已经加载
        return self.image_names  # 返回文件名列表

def load_low_wsi(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 应用转换
    image_tensor = transform(image)

    # 添加批次维度
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def load_high_wsi(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((2048, 2048)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 应用转换
    image_tensor = transform(image)

    # 添加批次维度
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def load_and_reshape_tensor(file_path):
    """
    加载.pt文件并改变Tensor的形状。

    参数:
    ----------
    file_path : str
        .pt文件的路径。
    new_shape : tuple
        新的Tensor形状。

    返回:
    -------
    reshaped_tensor : torch.Tensor
        改变形状后的Tensor。
    """
    # 加载.pt文件
    tensor = torch.load(file_path)
    new_shape = (1, 1, 1, 768)

    # 改变Tensor的形状
    reshaped_tensor = tensor.view(new_shape)

    return reshaped_tensor

# 使用示例
# reshaped_tensor = load_and_reshape_tensor('path_to_your_tensor.pt')

