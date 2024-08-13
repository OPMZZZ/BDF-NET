from data.base_dataset import BaseDataset
import os
import pydicom
import nibabel as nib
import numpy as np
import torch

time_array = [1, 43, 51, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
              90, 91, 92]


class DummyDataset(BaseDataset):
    """用于实现自定义数据集的模板数据集类。"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的数据集特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否是训练阶段。您可以使用此标志来添加训练特定或测试特定的选项。

        返回:
            修改后的解析器。
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='新数据集选项')
        parser.set_defaults(max_dataset_size=999, new_dataset_option=2.0)  # 指定数据集特定的默认值
        return parser

    def __init__(self, opt):
        """初始化数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志; 需要是BaseOptions的子类

        在这里可以做一些事情。
        - 保存选项（在BaseDataset中已完成）
        - 获取数据集的图像路径和元信息。
        - 定义图像转换。
        """
        # 保存选项和数据集根目录
        BaseDataset.__init__(self, opt)
        # 获取数据集中图像的路径；你可以调用sorted(make_dataset(self.root, opt.max_dataset_size))来获取目录self.root下所有图像的路径
        self.image_paths = []
        # 定义默认的转换函数。您可以使用<base_dataset.get_transform>；您也可以定义自己的自定义转换函数
        self.test_path = [os.path.join(opt.dataroot, x) for x in
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
        data = []
        for item in [x for x in os.listdir(os.path.join(data_dir, '92')) if
                     int(x.split('.')[0]) in time_array]:
            tmp = self.rescale(os.path.join(data_dir, '92', item))
            data.append(tmp)
        data = np.stack(data).astype(np.float32)
        rawdata = torch.from_numpy(data).permute(1, 0, 2, 3)
        data = torch.cat((rawdata[..., :6, :, :], rawdata[..., -6:, :, :]), dim=-3)
        label = rawdata[..., 6:-6, :, :]
        return (data, label)

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
        rescaled_data = rescale_slope * pixel_array + rescale_intercept

        return rescaled_data[..., 20:-20, 20:-20]
