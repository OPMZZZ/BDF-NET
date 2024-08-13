import os
from options.test_options import TestOptions
from data.myDataloader import create_dataset, CustomDataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import matplotlib.pyplot as plt


def draw(real, img1, img2, slice):
    img1 = img1.cpu().numpy()[0, [0, 5, 10, 15, -1], :, :]
    img2 = img2.cpu().numpy()[0, [0, 5, 10, 15, -1], :, :]
    real1 = real.cpu()[0, 0, :, :].unsqueeze(0).numpy()
    real2 = real.cpu()[0, -1, :, :].unsqueeze(0).numpy()
    img1 = np.concatenate((real1, img1, real2), axis=0)
    img2 = np.concatenate((real1, img2, real2), axis=0)
    img3 = np.abs(img1- img2)
    fig, axes = plt.subplots(3, 7, figsize=(26, 8))  # 2行7列
    imgs = [img1, img2]
    name = ['real', 'fake']
    # 假设 img1 和 img2 是要展示的多幅图像，可以将它们组织成列表或数组
    # 假设 imgs1 和 imgs2 是包含多幅图像的列表
    # 注意：确保 imgs1 和 imgs2 中包含足够多的图像以填充2行7列的子图
    for i in range(2):
        img = imgs[i]
        for j in range(7):
            img1 = img[j]
            # 在子图中显示 img1
            axes[i, j].imshow(img1, cmap='binary')  # 这里使用 'rainbow' colormap，根据需要更改
            # 隐藏坐标轴
            axes[i, j].axis('off')
    for j in range(7):
        im = axes[2, j].imshow(img3[j], cmap='jet')
        fig.colorbar(im, ax=axes[2, j])
        axes[2, j].axis('off')
    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(f'./outputs/{slice:03d}.png')
    plt.close()

if __name__ == '__main__':
    opt = TestOptions().parse()  # 获取测试选项
    # 为测试硬编码一些参数
    opt.num_threads = 0  # 测试代码仅支持 num_threads = 1
    opt.batch_size = 1  # 测试代码仅支持 batch_size = 1
    opt.serial_batches = True  # 禁用数据洗牌；如果需要在随机选择的图像上查看结果，请取消注释此行。
    opt.no_flip = True  # 无翻转；如果需要在翻转的图像上查看结果，请取消注释此行。
    opt.display_id = -1  # 无Visdom显示；测试代码将结果保存到HTML文件中。
    dataset = create_dataset(opt, size=1)  # 根据 opt.dataset_mode 和其他选项创建数据集
    model = create_model(opt)  # 根据 opt.model 和其他选项创建模型
    for i in range(1,2):
        opt.load_iter = i
        model.setup(opt)  # 常规设置：加载和打印网络；创建调度器
        # 创建一个网站
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义网站目录
        if opt.load_iter > 0:  # 默认情况下，load_iter 为0
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('创建网站目录', web_dir)
        webpage = html.HTML(web_dir, '实验 = %s，阶段 = %s，轮次 = %s' % (opt.name, opt.phase, opt.epoch))
        # 以 eval 模式进行测试。这仅影响像 batchnorm 和 dropout 等层。
        # 对于 [pix2pix]：我们在原始 pix2pix 中使用 batchnorm 和 dropout。您可以尝试带有和不带有 eval() 模式的效果。
        # 对于 [CycleGAN]：这不应影响 CycleGAN，因为 CycleGAN 使用没有 dropout 的 instancenorm。
        if opt.eval:
            model.eval()
        mean_rmse = []
        mean_ssim = []
        mean_psnr = []
        for _, patient in enumerate(dataset):  # 内循环，每个 epoch 内部

            minidataset = CustomDataset(patient)
            data_loader = torch.utils.data.DataLoader(minidataset, batch_size=opt.batch_size,
                                                      num_workers=opt.num_threads)
            for i, data in enumerate(data_loader):
                model.set_input(data)  # 从数据加载器中解压数据
                model.test()  # 进行推断
                visuals = model.get_current_visuals()  # 获取图像结果   # 获取图像路径
                x, y, z, d = model.cal_loss()
                mean_rmse.append(x)
                mean_ssim.append(y)
                mean_psnr.append(z)
                # draw(*d, i)
                # save_images(webpage, visuals, i, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            model.save()
            print(f'RMSE: {np.mean(mean_rmse)}')
            print(f'SSIM: {np.mean(mean_ssim)}')
            print(f'PSNR: {np.mean(mean_psnr)}')
        webpage.save()  # 保存HTML
