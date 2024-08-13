from .base_model import BaseModel
from . import mynetwork
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch
import numpy as np
import nibabel as nib
import os
import random
class TestModel(BaseModel):
    """ 这个TestModel可以用于仅处理一个方向的CycleGAN结果。
    这个模型将自动设置'--dataset_mode single'，该模式仅加载来自一个集合的图像。

    请参考测试说明以获取更多细节。
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重新编写现有选项的默认值。

        参数：
            parser（argparse.ArgumentParser） -- 原始选项解析器
            is_train（布尔值） -- 是否处于训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回：
            修改后的解析器。

        该模型只能在测试时使用。它需要'--dataset_mode single'。
        您需要使用选项'--model_suffix'指定网络。
        """
        assert not is_train, 'TestModel不能在训练时使用'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='_A',
                            help='在checkpoints_dir中，将加载[epoch]_net_G[model_suffix].pth作为生成器。')
        return parser

    def __init__(self, opt):
        """初始化pix2pix类。

        参数：
            opt（选项类） -- 存储所有实验标志的类；需要是BaseOptions的子类。
        """
        assert not opt.isTrain
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = []
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        self.visual_names = ['realB', 'fakeB']
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # 只需要生成器。
        self.netG = mynetwork.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                       opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.result = []
        self.origin = []
        # 将模型分配给self.netG_[suffix]，以便可以加载它
        # 请参阅<BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # 将netG存储在self中。

    def cal_loss(self):
        mse = torch.nn.MSELoss()
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        mean_nmse = []
        mean_ssim = []
        mean_psnr = []

        for i in range(self.realB.shape[1]):
            realb = self.realB[:, i]
            fakeb = self.fakeB[:, i]
            mean_nmse.append(mse(realb, fakeb).item() / (torch.mean(realb ** 2).item() + 1e-6))

            max_ = realb.max()
            realb /= max_+ 1e-6
            fakeb /= max_+ 1e-6

            psnr = PeakSignalNoiseRatio(data_range=1).to(self.device)
            mean_psnr.append(psnr(realb.unsqueeze(1), fakeb.unsqueeze(1)).item())

            mean_ssim.append(ssim(realb.unsqueeze(1), fakeb.unsqueeze(1)).item())
        max_ = max(self.realB.max(), self.fakeB.max(), self.real.max())
        return np.mean(mean_nmse), np.mean(mean_ssim), np.mean(mean_psnr), (
        self.real / max_+ 1e-6, self.realB / max_+ 1e-6, self.fakeB / max_+ 1e-6)

    def set_input(self, input):
        """从数据加载器解包输入数据并执行必要的预处理步骤。

        参数：
            input（字典） -- 包含数据本身及其元数据信息的字典。

        我们需要使用'single_dataset'数据集模式。它只加载来自一个域的图像。
        """
        self.real = input['A'].to(self.device)
        self.realB = input['B'].to(self.device)

    def forward(self):
        """运行前向传递。"""
        self.fakeB = self.netG(self.real)  # G(real)
        self.result.append(torch.cat((self.real[:, :6], self.fakeB, self.real[:, -6:]), dim=1))
        self.origin.append(torch.cat((self.real[:, :6], self.realB, self.real[:, -6:]), dim=1))



    def save(self):
        pn = self.opt.patient_name.split("\\")[-1]
        print(pn)
        if not os.path.exists(os.path.join('body',self.opt.name)):
            os.mkdir(os.path.join('body',self.opt.name))
        path = os.path.join('body',self.opt.name, pn)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'real'))
            os.mkdir(os.path.join(path, 'predict'))
        result = torch.cat(self.result, dim=0).cpu().numpy()
        for i in range(result.shape[1]):
            nifti_image = nib.Nifti1Image(result[:, i], affine=np.eye(4))
            nib.save(nifti_image, os.path.join(path, 'predict', f'{i + 1:08d}.nii.gz'))

        self.result = []


        origin = torch.cat(self.origin, dim=0).cpu().numpy()
        for i in range(origin.shape[1]):
            nifti_image = nib.Nifti1Image(origin[:, i], affine=np.eye(4))
            nib.save(nifti_image, os.path.join(path, 'real', f'{i + 1:08d}.nii.gz'))
        self.origin = []


    def optimize_parameters(self):
        """对于测试模型，不执行优化。"""
        pass
