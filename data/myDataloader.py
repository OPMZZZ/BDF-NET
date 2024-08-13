from __future__ import print_function, division
import os
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import h5py
from tqdm import tqdm

time_array = [1, 43, 51, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
              90, 91, 92]

time_duration = [2] * 30 + [5] * 12 + [10] * 6 + [30] * 4 + [60] * 25 + [120] * 15


# class DummyDataset(Dataset):D
#     """用于实现自定义数据集的模板数据集类。"""
#
#     def __init__(self, opt):
#         """初始化数据集类。
#
#         参数:
#             opt (Option类) -- 存储所有实验标志; 需要是BaseOptions的子类
#
#         在这里可以做一些事情。
#         - 保存选项（在BaseDataset中已完成）
#         - 获取数据集的图像路径和元信息。
#         - 定义图像转换。
#         """
#         # 获取数据集中图像的路径；你可以调用sorted(make_dataset(self.root, opt.max_dataset_size))来获取目录self.root下所有图像的路径
#         self.image_paths = []
#         # 定义默认的转换函数。您可以使用<base_dataset.get_transform>；您也可以定义自己的自定义转换函数
#         self.test_path = [os.path.join(opt.dataroot, x) for x in
#                           open(os.path.join(opt.dataroot, '1.txt'), 'r').readlines()]
#         self.train_path = [os.path.join(opt.dataroot, x) for x in os.listdir(opt.dataroot) if not x.endswith('txt')
#                            if
#                            os.path.join(opt.dataroot, x) not in self.test_path]
#         self.now_path = self.train_path if opt.is_train else self.test_path
#
#     def __getitem__(self, index):
#         """返回一个数据点和其元数据信息。
#
#         参数:
#             index -- 用于数据索引的随机整数
#
#         返回:
#             一个包含数据及其名称的字典。通常包含数据本身和其元数据信息。
#
#         步骤1: 获取一个随机图像路径，例如path = self.image_paths[index]
#         步骤2: 从磁盘加载数据，例如image = Image.open(path).convert('RGB')。
#         步骤3: 将数据转换为PyTorch张量。您可以使用辅助函数，例如self.transform，例如data = self.transform(image)。
#         步骤4: 返回一个数据点，作为一个字典。
#         """
#
#         data_dir = self.now_path[index]
#         data = []
#         time_list = [x for x in os.listdir(os.path.join(data_dir, '92')) if
#                      int(x.split('.')[0]) in time_array]
#         time_list.sort(key=lambda x: int(x.split('.')[0]))
#         for item in tqdm(time_list):
#             tmp = self.rescale(os.path.join(data_dir, '92', item))
#             data.append(tmp)
#         data = np.stack(data).astype(np.float32)
#         rawdata = torch.from_numpy(data).permute(1, 0, 2, 3)
#         data = torch.cat((rawdata[..., :6, :, :], rawdata[..., -6:, :, :]), dim=-3)
#         label = rawdata[..., 6:-6, :, :]
#         return (data, label)
#
#     def __len__(self):
#         """返回图像的总数。"""
#         return len(self.now_path)
#
#     def rescale(self, file_path):
#         # 读取DICOM文件
#         ds = pydicom.dcmread(file_path)
#         # 获取Rescale Slope和Rescale Intercept
#         rescale_slope = ds.get("RescaleSlope", 1.0)  # 默认为1.0
#         rescale_intercept = ds.get("RescaleIntercept", 0.0)  # 默认为0.0
#         pixel_array = ds.pixel_array
#         # 根据公式计算每个像素点的值
#
#         pixel_array = pixel_array[80:560, 20:-20, 20:-20]
#         rescaled_data = rescale_slope * pixel_array + rescale_intercept
#         # return rescaled_data[80:660, 20:-20, 20:-20]
#         return rescaled_data

class DummyDataset(Dataset):
    """用于实现自定义数据集的模板数据集类。"""

    def __init__(self, opt):
        """初始化数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志; 需要是BaseOptions的子类

        在这里可以做一些事情。
        - 保存选项（在BaseDataset中已完成）
        - 获取数据集的图像路径和元信息。
        - 定义图像转换。
        """
        # 获取数据集中图像的路径；你可以调用sorted(make_dataset(self.root, opt.max_dataset_size))来获取目录self.root下所有图像的路径
        self.image_paths = []
        self.opt = opt
        # 定义默认的转换函数。您可以使用<base_dataset.get_transform>；您也可以定义自己的自定义转换函数
        self.test_path = [os.path.join(opt.dataroot, x.strip()) for x in
                          open(os.path.join(opt.dataroot, '1.txt'), 'r').readlines()]
        self.train_path = [os.path.join(opt.dataroot, x) for x in os.listdir(opt.dataroot) if not x.endswith('txt')
                           if
                           os.path.join(opt.dataroot, x) not in self.test_path]
        self.now_path = self.train_path if opt.is_train else self.test_path

    def __getitem__(self, index):
        """返回一个数据点和其元数据信息。

        参数:
            index -- 用于数据索引的随机整数

        返回:
            一个包含数据及其名称的字典。通常包含数据本身和其元数据信息。

        步骤1: 获取一个随机图像路径，例如path = self.image_paths[index]
        步骤2: 从磁盘加载数据，例如image = Image.open(path).convert('RGB')。
        步骤3: 将数据转换为PyTorch张量。您可以使用辅助函数，例如self.transform，例如data = self.transform(image)。
        步骤4: 返回一个数据点，作为一个字典。
        """

        data_dir = self.now_path[index]
        self.opt.patient_name = data_dir.split("/")[-1]
        rawdata = []
        mat_data = h5py.File(os.path.join(data_dir, 'DynamicPET.mat'), 'r')
        w, m = [x.strip() for x in open(os.path.join(data_dir, 'info.txt'), 'r').readlines()]
        data = mat_data['X'][:].transpose(1, 0, 3, 2)  # 替换为MAT文件中的图像数据字段名称
        for i in tqdm(time_array):
            sum_duration = 0
            i -= 1
            # if i < 43:
            index = i
            tmp = np.zeros_like(data[:, 0])
            while sum_duration < 120:
                tmp += data[:, index] * time_duration[index] / 120
                sum_duration += time_duration[index]
                index += 1

            assert sum_duration == 120
            # else:
            #     tmp = data[:, i]
            rawdata.append(tmp)

        # rawdata = np.stack(rawdata, axis=1).astype(np.float32)[80:560]
        rawdata = np.stack(rawdata, axis=1).astype(np.float32)[300:400]
        rawdata = torch.from_numpy(rawdata)
        rawdata = rawdata / (float(m) * 3.7e7 / (1000 * float(w)))
        return rawdata
    def __len__(self):
        """返回图像的总数。"""
        return len(self.now_path)

    def rescale(self, file_path):
        # 读取DICOM文件
        ds = pydicom.dcmread(file_path)
        # 获取Rescale Slope和Rescale Intercept
        rescale_slope = ds.get("RescaleSlope", 1.0)  # 默认为1.0
        rescale_intercept = ds.get("RescaleIntercept", 0.0)  # 默认为0.0
        pixel_array = ds.pixel_array
        # 根据公式计算每个像素点的值

        pixel_array = pixel_array[80:560, 20:-20, 20:-20]
        rescaled_data = rescale_slope * pixel_array + rescale_intercept
        # return rescaled_data[80:660, 20:-20, 20:-20]
        return rescaled_data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data.view(-1, 30, 192, 192)
        # self.transform = transforms.Compose([transforms.RandomHorizontalFlip()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        # rawdata = (rawdata - torch.mean(rawdata)) / torch.std(rawdata)
        # data = (rawdata - rawdata.min()) / (rawdata.max() - rawdata.min()) * 50
        data_sample = torch.cat((data[:6], data[-6:]))
        label_sample = data[6:-6]

        # data_A = self.transform(data_A)

        # data_B = self.transform(data_B)
        return {'A': data_sample, 'B': label_sample}


def create_dataset(opt, size=4):
    dataloader = DataLoader(DummyDataset(opt), batch_size=size)
    return dataloader
