
import time
from options.train_options import TrainOptions
from data.myDataloader import create_dataset, CustomDataset
from models import create_model
from util.visualizer import Visualizer
import torch
from tqdm import tqdm
if __name__ == '__main__':
    opt = TrainOptions().parse()  # 获取训练选项
    dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他选项创建数据集
    dataset_size = len(dataset)*4  # 获取数据集中图像的数量
    print('训练病人的数量 = %d 训练图像总数%d' % (dataset_size, dataset_size*560))
    model = create_model(opt)  # 根据 opt.model 和其他选项创建模型
    model.setup(opt)  # 常规设置：加载和打印网络；创建调度器
    visualizer = Visualizer(opt)  # 创建一个可显示/保存图像和绘图的可视化工具
    total_iters = 0  # 总的训练迭代次数

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # 外循环，用于不同的 epoch；我们通过 <epoch_count>, <epoch_count>+<save_latest_freq> 来保存模型
        epoch_start_time = time.time()  # 记录整个 epoch 的时间
        iter_data_time = time.time()  # 记录数据加载的时间
        epoch_iter = 0  # 当前 epoch 中的训练迭代次数，每个 epoch 重置为 0
        visualizer.reset()  # 重置可视化工具：确保每个 epoch 至少将结果保存到 HTML 一次
        model.update_learning_rate()  # 每个 epoch 开始时更新学习率
        for _, patient in enumerate(dataset):  # 内循环，每个 epoch 内部

            minidataset = CustomDataset(patient)
            data_loader = torch.utils.data.DataLoader(minidataset, batch_size=opt.batch_size, shuffle=True,
                                                      num_workers=opt.num_threads)
            for i in range(1):
                for data in data_loader:
                    iter_start_time = time.time()  # 记录每次迭代的计算时间
                    if total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)  # 从数据集中解包数据并应用预处理
                    model.optimize_parameters(i)  # 计算损失函数，获取梯度，更新网络权重

                    if total_iters % opt.display_freq == 0:  # 在 visdom 上显示图像并保存图像到 HTML 文件
                        save_result = total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_iters % opt.print_freq == 0:  # 打印训练损失并将日志信息保存到磁盘
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(total_iters) / dataset_size, losses)

                    if total_iters % opt.save_latest_freq == 0:  # 每 <save_latest_freq> 次迭代时缓存我们的最新模型
                        print('保存最新模型 (epoch %d, total_iters %d)' % (epoch, total_iters))
                        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)
                iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # 每 <save_epoch_freq> 个 epoch 时缓存我们的模型
            print('在 epoch %d 结束时保存模型，迭代次数 %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('结束 epoch %d / %d \t 花费时间: %d 秒' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
