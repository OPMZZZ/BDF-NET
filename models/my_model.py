import random

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import mynetwork
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio



class mymodel(BaseModel):
    """
    这个类实现了CycleGAN模型，用于学习无配对数据的图像到图像的翻译。

    模型训练需要使用'--dataset_mode unaligned'数据集。
    默认情况下，它使用一个'--netG resnet_9blocks'的ResNet生成器，
    一个'--netD basic'鉴别器（由pix2pix引入的PatchGAN），
    以及最小二乘GANs目标函数（'--gan_mode lsgan'）。

    CycleGAN论文：https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重新编写现有选项的默认值。

        参数：
            parser（argparse.ArgumentParser） -- 原始选项解析器
            is_train（布尔值） -- 是否处于训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回：
            修改后的解析器。

        对于CycleGAN，除了GAN损失，我们引入lambda_A、lambda_B和lambda_identity用于以下损失。
        A（源领域），B（目标领域）。
        生成器：G_A: A -> B; G_B: B -> A。
        鉴别器：D_A: G_A(A) vs. B; D_B: G_B(B) vs. A。
        前向循环损失：lambda_A * ||G_B(G_A(A)) - A||（论文中的公式（2））
        后向循环损失：lambda_B * ||G_A(G_B(B)) - B||（论文中的公式（2））
        身份损失（可选）：lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)（第5.2节“从绘画生成照片”中的内容）
        原始CycleGAN论文中没有使用Dropout。
        """
        parser.set_defaults(no_dropout=True)  # 默认情况下，CycleGAN不使用Dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1, help='循环损失的权重（A -> B -> A）')
            parser.add_argument('--lambda_B', type=float, default=1, help='循环损失的权重（B -> A -> B）')
            parser.add_argument('--lambda_identity', type=float, default=0,
                                help='使用身份映射。将lambda_identity设置为非零会影响身份映射损失的权重。例如，如果身份损失的权重应该比重构损失的权重小10倍，请将lambda_identity设置为0.1')

        return parser

    def __init__(self, opt):
        """初始化CycleGAN类。

        参数：
            opt（选项类） -- 存储所有实验标志的类；需要是BaseOptions的子类。
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['DD_A', 'GG_A', 'c_A', "f_G_A", 's_G_A', 'p_G_A', 'DD_B', 'GG_B', 'c_B', "f_G_B",
                           's_G_B',
                           'p_G_B']
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # 如果使用身份损失，我们还会可视化idt_B=G_A(B)和idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # self.visual_names = visual_names_A + visual_names_B  # 将A和B的可视化组合在一起
        self.visual_names = ['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B', 'diff_A',
                             'diff_B']  # 将A和B的可视化组合在一起
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_mynetwork>和<BaseModel.load_mynetwork>。
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # 在测试时，只加载Gs
            self.model_names = ['G_A', 'G_B']

        self.loss_GG_A = []
        self.loss_GG_B = []
        self.loss_DD_A = []
        self.loss_DD_B = []
        self.loss_c_A = []
        self.loss_f_G_A = []
        self.loss_p_G_A = []
        self.loss_s_G_A = []
        self.loss_c_B = []
        self.loss_f_G_B = []
        self.loss_p_G_B = []
        self.loss_s_G_B = []
        # 定义网络（包括生成器和鉴别器）
        # 命名与论文中使用的不同。
        # 代码（与论文对比）：G_A（G）、G_B（F）、D_A（D_Y）、D_B（D_X）
        self.netG_A = mynetwork.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = mynetwork.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, reverse=True)

        if self.isTrain:  # 定义鉴别器
            self.netD_A = mynetwork.define_D(opt.dis_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_B = mynetwork.define_D(opt.dis_nc, opt.ndf, opt.netD,
            #                                  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
            #                                  reverse=True)
            self.netD_B = self.netD_A
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # 仅当输入和输出图像具有相同通道数时才有效
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储先前生成的图像
            self.fake_B_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储先前生成的图像
            # 定义损失函数
            self.criterionGAN = mynetwork.GANLoss(opt.gan_mode).to(self.device)  # 定义GAN损失。
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.SSIM = StructuralSimilarityIndexMeasure().to(self.device)
            # 初始化优化器；调度器将由函数<BaseModel.setup>自动创建。
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从数据加载器解包输入数据并执行必要的预处理步骤。

        参数：
            input（字典） -- 包含数据本身及其元数据信息的字典。

        选项'direction'可用于交换领域A和领域B。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """运行前向传递；由函数<optimize_parameters>和<test>调用。"""
        #
        # noise_A = 0.1 * torch.randn(self.real_A.size()).to(self.device)
        # noise_B = 0.1 * torch.randn(self.real_B.size()).to(self.device)
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def myCat(self, a, b):
        ch = a.shape[-3] // 2
        x0 = a[..., :ch, :, :]
        x1 = a[..., ch:, :, :]
        x = torch.cat((x0, b, x1), dim=-3)
        return x

    def backward_D_basic(self, netD, reala, realb, fake, flag):
        """计算鉴别器的GAN损失

        参数：
            netD (network)      -- 鉴别器D
            real (tensor array) -- 真实图像
            fake (tensor array) -- 由生成器生成的图像

        返回鉴别器损失。
        我们还调用loss_D.backward()来计算梯度。
        """
        real = self.myCat(reala, realb)
        if flag == 'a':
            fake = self.myCat(fake.detach(), realb)
        else:
            fake = self.myCat(reala, fake.detach())

        c = real.shape[1]
        true_time = list(range(c))
        fake_time = true_time.copy()
        while fake_time == true_time:
            random.shuffle(fake_time)
        random_real = real[:, fake_time, :, :]

        pred_real = netD(real)
        # 真实图像(random
        pred_random = netD(random_real)
        # 生成图像
        pred_fake = netD(fake)

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_random = self.criterionGAN(pred_random, False)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # 组合损失并计算梯度
        loss_D = (loss_D_real + loss_D_fake + loss_D_random) * 0.333
        loss_D.backward()
        return loss_D

    def make_random(self, real):

        c = real.shape[1]
        true_time = list(range(c))
        fake_time = true_time.copy()
        while fake_time == true_time:
            random.shuffle(fake_time)
        random_real = real[:, fake_time, :, :]
        flag = [true_time[i] != fake_time[i] for i in range(c)]

        return random_real, flag

    def backward_D_A(self):
        """计算鉴别器D_A的GAN损失"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.real_B, fake_B, 'b')
        self.loss_DD_A.append(self.loss_D_A.item())

    def backward_D_B(self):
        """计算鉴别器D_B的GAN损失"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.real_B, fake_A, 'a')
        self.loss_DD_B.append(self.loss_D_B.item())

    def backward_G(self):
        """计算生成器G_A和G_B的损失"""
        lambda_A = 2
        lambda_B = 2

        # GAN损失 D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.myCat(self.real_A, self.fake_B)), True)
        # GAN损失 D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.myCat(self.fake_A, self.real_B)), True)
        # 前向循环损失 || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # 后向循环损失 || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # 组合损失并计算梯度
        self.loss_f1_G_A = self.criterionCycle(self.real_B, self.fake_B)
        self.loss_f1_G_B = self.criterionCycle(self.real_A, self.fake_A)

        # random_realb, flag = self.make_random(self.real_B)
        # c = self.real_B.shape[1]
        # loss = 0
        # for i in range(c):
        #     loss += (-1) ** flag[i] * self.criterionCycle(random_realb[:, i], self.fake_B[:, i]) * (0.3 + 0.01 * i)
        # lossa = loss / c
        #
        # random_reala, flag = self.make_random(self.real_A)
        # c = self.real_A.shape[1]
        # loss = 0
        # for i in range(self.real_A.shape[1]):
        #     loss += (-1) ** flag[i] * self.criterionCycle(random_reala[:, i], self.fake_A[:, i]) * (0.3 + 0.01 * i)
        # lossb = loss / c
        self.loss_ssim_G_A = self.SSIM(self.fake_B, self.real_B)
        self.loss_ssim_G_B = self.SSIM(self.fake_A, self.real_A)

        self.loss_psnr_G_A = 0
        self.loss_psnr_G_B = 0
        tb = self.fake_B.shape[1]
        for i in range(tb):
            PSNR = PeakSignalNoiseRatio(data_range=self.real_B[0, i].max().item() + 1e-6).to(self.device)
            self.loss_psnr_G_A += PSNR(self.fake_B[0, i], self.real_B[0, i])
        ta = self.fake_A.shape[1]
        for i in range(ta):
            PSNR = PeakSignalNoiseRatio(data_range=self.real_A[0, i].max().item() + 1e-6).to(self.device)
            self.loss_psnr_G_B += PSNR(self.fake_A[0, i], self.real_A[0, i])
        self.loss_psnr_G_A /= tb
        self.loss_psnr_G_B /= ta

        self.loss_G = self.loss_G_A + self.loss_G_B + (
                1 - self.loss_ssim_G_A) + (1 - self.loss_ssim_G_B) + (
                              self.loss_f1_G_A + self.loss_f1_G_B + self.loss_cycle_A + self.loss_cycle_B) * 0.005

        self.loss_GG_A.append(self.loss_G_A.item())
        self.loss_GG_B.append(self.loss_G_B.item())

        self.loss_c_A.append(self.loss_cycle_A.item())
        self.loss_c_B.append(self.loss_cycle_B.item())
        self.loss_f_G_A.append(self.loss_f1_G_A.item())
        self.loss_f_G_B.append(self.loss_f1_G_B.item())
        self.loss_p_G_A.append(self.loss_psnr_G_A.item())
        self.loss_p_G_B.append(self.loss_psnr_G_B.item())
        self.loss_s_G_A.append(self.loss_ssim_G_A.item())
        self.loss_s_G_B.append(self.loss_ssim_G_B.item())

        self.loss_G.backward()

    def optimize_parameters(self, i):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用"""
        for _ in range(1):
            # 前向传递
            self.forward()  # 计算生成的图像和重构图像。
            # G_A和G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # 在优化G时，D不需要梯度
            self.optimizer_G.zero_grad()  # 将G_A和G_B的梯度设置为零
            self.backward_G()  # 计算G_A和G_B的梯度
            self.optimizer_G.step()  # 更新G_A和G_B的权重

        # D_A和D_B
        if i % 2 == 0:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # 将D_A和D_B的梯度设置为零
            self.backward_D_A()  # 计算D_A的梯度
            self.backward_D_B()  # 计算D_B的梯度
            self.optimizer_D.step()  # 更新D_A和D_B的权重

        self.diff_A = torch.abs(self.real_A - self.fake_A.detach())
        self.diff_B = torch.abs(self.real_B - self.fake_B.detach())
