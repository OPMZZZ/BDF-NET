import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.DIY_net import AttU_Net



###############################################################################
# 辅助函数
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """返回一个标准化层

    参数:
        norm_type (str) -- 标准化层的名称: batch | instance | none

    对于BatchNorm，我们使用可学习的仿射参数并跟踪运行统计数据（均值/标准差）。
    对于InstanceNorm，我们不使用可学习的仿射参数，也不跟踪运行统计数据。
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('标准化层 [%s] 未找到' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """返回一个学习率调度器

    参数:
        optimizer          -- 网络的优化器
        opt (option class) -- 存储所有实验标志的类; 需要是BaseOptions的子类
                              opt.lr_policy是学习率策略的名称: linear | step | plateau | cosine

    对于'linear'，我们在前opt.n_epochs个时期内保持相同的学习率，
    然后线性降低学习率到零，这需要额外的opt.n_epochs_decay个时期。
    对于其他调度器（step, plateau和cosine），我们使用PyTorch的默认调度器。
    请参阅https://pytorch.org/docs/stable/optim.html以获取更多详细信息。
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('学习率策略 [%s] 未实现', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重

    参数:
        net (network)   -- 要初始化的网络
        init_type (str) -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- normal、xavier和orthogonal的缩放因子。

    我们在原始的pix2pix和CycleGAN论文中使用'normal'。但是xavier和kaiming可能
    对某些应用效果更好。可以自行尝试。
    """

    def init_func(m):  # 定义初始化函数
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('初始化方法 [%s] 未实现' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm层的权重不是矩阵；只有normal分布适用。
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('使用 %s 初始化网络' % init_type)
    net.apply(init_func)  # 应用初始化函数<init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """初始化网络: 1. 注册CPU/GPU设备（支持多GPU）; 2. 初始化网络权重
    参数:
        net (network)      -- 要初始化的网络
        init_type (str)    -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        gain (float)       -- normal、xavier和orthogonal的缩放因子
        gpu_ids (int list) -- 网络在哪些GPU上运行: 例如，0,1,2

    返回一个初始化后的网络。
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # 多GPU
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[], reverse=False):
    """创建生成器

    参数:
        input_nc (int) -- 输入通道数
        output_nc (int) -- 输出通道数
        ngf (int) -- 生成器的特征映射数
        netG (str) -- 网络名称: resnet | unet | resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- 标准化层的名称: batch | instance | none
        use_dropout (bool) -- 是否使用Dropout层
        init_type (str)    -- 初始化方法的名称
        init_gain (float)  -- 初始化缩放因子
        gpu_ids (int list) -- 网络在哪些GPU上运行: 例如，0,1,2

    返回一个初始化后的生成器
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = DoubleResNetGenerator(input_nc, output_nc, ngf, reverse, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, reverse, norm_layer=norm_layer, use_dropout=use_dropout,
                              n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('生成器 [%s] 未实现' % net)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[],
             reverse=False):
    """创建判别器

    参数:
        input_nc (int) -- 输入通道数
        ndf (int) -- 判别器的特征映射数
        netD (str) -- 网络名称: basic | n_layers | pixel
        n_layers_D (int) -- 判别器的层数
        norm (str) -- 标准化层的名称: batch | instance | none
        init_type (str) -- 初始化方法的名称
        init_gain (float)    -- 初始化缩放因子
        gpu_ids (int list) -- 网络在哪些GPU上运行: 例如，0,1,2

    返回一个初始化后的判别器
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, reverse, n_layers=5, norm_layer=norm_layer)
    elif netD == 'n_layers':
        net = MyProjectDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('判别器 [%s] 未实现' % netD)

    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# 类
##############################################################################
class GANLoss(nn.Module):
    """定义不同的GAN目标函数。

    GANLoss类抽象了创建与输入相同大小的目标标签张量的需求。
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ 初始化GANLoss类。

        参数:
            gan_mode (str) - - GAN目标函数的类型。目前支持vanilla、lsgan和wgangp。
            target_real_label (bool) - - 真实图像的标签
            target_fake_label (bool) - - 伪造图像的标签

        注意: 不要在鉴别器的最后一层使用sigmoid函数。
        LSGAN不需要sigmoid函数。vanilla GAN会使用BCEWithLogitsLoss处理它。
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('GAN模式 %s 未实现' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """创建与输入相同大小的标签张量。

        参数:
            prediction (tensor) - - 通常是来自鉴别器的预测
            target_is_real (bool) - - 地面真实标签是否是真实图像或伪造图像

        返回:
            一个填充有地面真实标签的标签张量，与输入大小相同
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """根据鉴别器的输出和地面真实标签计算损失。

        参数:
            prediction (tensor) - - 通常是鉴别器的输出预测
            target_is_real (bool) - - 地面真实标签是否是真实图像或伪造图像

        返回:
            计算得到的损失。
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """计算梯度惩罚损失，用于WGAN-GP论文 https://arxiv.org/abs/1704.00028

    参数:
        netD (网络)              -- 鉴别器网络
        real_data (张量数组)    -- 真实图像
        fake_data (张量数组)    -- 生成器生成的伪造图像
        device (str)            -- GPU / CPU: 来自torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)              -- 是否混合使用真实和伪造数据 [real | fake | mixed].
        constant (float)        -- 在公式中使用的常数 ( ||gradient||_2 - constant)^2
        lambda_gp (float)       -- 此损失的权重

    返回梯度惩罚损失
    """
    if lambda_gp > 0.0:
        if type == 'real':  # 使用真实图像、伪造图像，或两者的线性插值。
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} 未实现'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # 拉平数据
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # 添加eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """基于ResNet的生成器，由一些下采样/上采样操作之间的ResNet块组成。

    我们借鉴了Justin Johnson的神经风格迁移项目（https://github.com/jcjohnson/fast-neural-style）中的Torch代码和思路。
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """构建一个基于ResNet的生成器

        参数:
            input_nc (int)      -- 输入图像中的通道数
            output_nc (int)     -- 输出图像中的通道数
            ngf (int)           -- 最后一个卷积层中的滤波器数量
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层
            n_blocks (int)      -- ResNet块的数量
            padding_type (str)  -- 卷积层中填充层的名称: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # 添加下采样层
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 添加ResNet块

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # 添加上采样层
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """标准前向传播"""
        return self.model(input)


class DoubleResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, reverse, norm_layer, use_dropout, n_blocks,
                 ):
        assert (n_blocks >= 0)
        super(DoubleResNetGenerator, self).__init__()
        self.reverse = reverse
        self.output_nc = output_nc
        if not reverse:
            input_nc += 1
            output_nc *= 2
        # self.model0 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type)
        # self.model1 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type)
        self.model0 = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.model1 = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # self.model0 = residual_transformers.ResViT(transformer_configs.get_resvit_b16_config(), input_dim=input_nc, img_size=192,
        #                                            output_dim=output_nc, vis=False)
        # self.model1 = residual_transformers.ResViT(transformer_configs.get_resvit_b16_config(), input_dim=input_nc, img_size=192,
        #                                            output_dim=output_nc, vis=False)
        # self.model1 = AttU_Net(input_nc, output_nc)

    def forward(self, x):
        """
        标准前向传播
        a → b1       reverse_c → reverse_b2
        b2 → c       reverse_b1 → reverse_a

        """
        inch = x.shape[-3] // 2
        if self.reverse:
            x0 = x[..., inch:, :, :]
            x1 = x[..., :inch, :, :]
        else:
            x0 = x[..., :inch, :, :]
            x1 = x[..., inch:, :, :]
        x1 = x1[..., list(range(inch))[::-1], :, :]

        # 补一张
        if not self.reverse:
            patch_1 = x1[:, -1, :, :].unsqueeze(1)
            patch_0 = x0[:, -1, :, :].unsqueeze(1)
            x0 = torch.cat((x0, patch_1), dim=1)
            x1 = torch.cat((x1, patch_0), dim=1)

        y0 = self.model0(x0)

        y1 = self.model1(x1)
        outch = y1.shape[-3]
        y1 = y1[..., list(range(outch))[::-1], :, :]
        #补一张
        if not self.reverse:
            out0 = y0[:, :outch//2, :, :]*0.9 + y1[:, :outch//2, :, :] * 0.1
            out1 = y1[:, outch//2:, :, :]*0.9 + y0[:, outch//2:, :, :] * 0.1
            y0 = out0
            y1 = out1
        if self.reverse:
            y = torch.cat((y1, y0), dim=-3)
        else:
            y = torch.cat((y0, y1), dim=-3)
        return y


class ResnetBlock(nn.Module):
    """定义一个ResNet块"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化ResNet块

        ResNet块是一个带有跳跃连接的卷积块
        我们使用build_conv_block函数构建卷积块，
        并在<forward>函数中实现跳跃连接。
        原始的ResNet论文: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构建一个卷积块。

        参数:
            dim (int)           -- 卷积层中的通道数。
            padding_type (str)  -- 填充层的名称: reflect | replicate | zero
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层。
            use_bias (bool)     -- 卷积层是否使用偏差

        返回一个卷积块（包括卷积层、标准化层和非线性层（ReLU））
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('填充 [%s] 未实现' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('填充 [%s] 未实现' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """前向传播函数（带有跳跃连接）"""
        out = x + self.conv_block(x)  # 添加跳跃连接
        return out


class UnetGenerator(nn.Module):
    """创建一个基于Unet的生成器"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构建Unet生成器
        参数:
            input_nc (int)  -- 输入图像中的通道数
            output_nc (int) -- 输出图像中的通道数
            num_downs (int) -- UNet中的下采样次数。例如，如果 |num_downs| == 7，
                                那么大小为128x128的图像将在瓶颈处变为1x1。
            ngf (int)       -- 最后一个卷积层中的滤波器数量
            norm_layer      -- 标准化层

        我们从内层到外层构建U-Net。
        这是一个递归过程。
        """
        super(UnetGenerator, self).__init__()
        # 构建unet结构
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # 添加最内层
        for i in range(num_downs - 5):  # 添加具有ngf * 8滤波器的中间层
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # 逐渐减少滤波器数量从ngf * 8到ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # 添加最外层

    def forward(self, input):
        """标准前向传播"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """定义具有跳跃连接的Unet子模块。
        X -------------------标识----------------------
        |-- 下采样 -- |子模块| -- 上采样 --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构建具有跳跃连接的Unet子模块。

        参数:
            outer_nc (int) -- 外部卷积层中的滤波器数量
            inner_nc (int) -- 内部卷积层中的滤波器数量
            input_nc (int) -- 输入图像/特征中的通道数
            submodule (UnetSkipConnectionBlock) -- 先前定义的子模模块
            outermost (bool)    -- 如果此模块是最外层模块
            innermost (bool)    -- 如果此模块是最内层模块
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层。
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        sig = nn.Sigmoid()
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, uprelu]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.outermost:
            x = self.model(x)
            x = self.relu(x)
            return x
        else:  # 添加跳跃连接
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """定义一个PatchGAN鉴别器"""

    def __init__(self, input_nc, ndf, reverse, n_layers=3, norm_layer=nn.BatchNorm2d):
        """构建一个PatchGAN鉴别器

        参数:
            input_nc (int)  -- 输入图像中的通道数
            ndf (int)       -- 最后一个卷积层中的滤波器数量
            n_layers (int)  -- 鉴别器中的卷积层数量
            norm_layer      -- 标准化层
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # 不需要使用偏差，因为BatchNorm2d具有仿射参数
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐渐增加滤波器数量
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # 输出1通道预测图
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """标准前向传播。"""

        return self.model(x)


class PixelDiscriminator(nn.Module):
    """定义一个1x1 PatchGAN鉴别器（像素GAN）"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """构建一个1x1 PatchGAN鉴别器

        参数:
            input_nc (int)  -- 输入图像中的通道数
            ndf (int)       -- 最后一个卷积层中的滤波器数量
            norm_layer      -- 标准化层
        """
        super(PixelDiscriminator, self).__init()
        if type(norm_layer) == functools.partial:  # 不需要使用偏差，因为BatchNorm2d具有仿射参数
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """标准前向传播。"""
        return self.net(input)


class MyProjectDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, reverse, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(MyProjectDiscriminator, self).__init__()
        self.model = AttU_Net(input_nc)
        self.reverse = reverse

    def forward(self, x):
        """标准前向传播。"""

        return self.model(x)


if __name__ == '__main__':
    net = DoubleResNetGenerator(6, 9).cuda()
    x = torch.rand([1, 18, 320, 320]).cuda()
    y = torch.rand([1, 12, 320, 320]).cuda()
    x = net(x, y)
    print()
