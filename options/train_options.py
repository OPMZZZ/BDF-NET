from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--is_train', type=bool, default=True, help='是否为训练模式')
        parser.add_argument('--display_freq', type=int, default=200, help='在屏幕上显示训练结果的频率')
        parser.add_argument('--display_ncols', type=int, default=3, help='如果为正数，将所有图像显示在单个Visdom网络面板中，每行显示一定数量的图像')
        parser.add_argument('--display_id', type=int, default=1, help='Web显示窗口的ID')
        parser.add_argument('--display_server', type=str, default="http://127.0.0.1", help='Visdom网络显示的服务器')
        parser.add_argument('--display_env', type=str, default='main', help='Visdom显示环境名称（默认为“main”）')
        parser.add_argument('--display_port', type=int, default=8888, help='Visdom网络显示的端口')
        parser.add_argument('--update_html_freq', type=int, default=673, help='保存训练结果到HTML的频率')
        parser.add_argument('--print_freq', type=int, default=100, help='在控制台上显示训练结果的频率')
        parser.add_argument('--no_html', action='store_true', help='不将中间训练结果保存到[opt.checkpoints_dir]/[opt.name]/web/')
        # 网络保存和加载参数
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='保存最新结果的频率')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='在每个周期结束时保存检查点的频率')
        parser.add_argument('--save_by_iter', action='store_true', help='是否按迭代保存模型')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：加载最新的模型')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='起始周期计数，我们通过<epoch_count>，<epoch_count>+<save_latest_freq>等方式保存模型')
        parser.add_argument('--phase', type=str, default='train', help='训练、验证、测试等阶段')
        # 训练参数
        parser.add_argument('--n_epochs', type=int, default=3, help='具有初始学习率的周期数')
        parser.add_argument('--n_epochs_decay', type=int, default=5, help='将学习率线性衰减至零的周期数')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam的动量项')
        parser.add_argument('--lr', type=float, default=0.0006, help='Adam的初始学习率')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='GAN目标的类型。[vanilla|lsgan|wgangp]。vanilla GAN loss是原始GAN论文中使用的交叉熵目标。')
        parser.add_argument('--pool_size', type=int, default=50, help='存储先前生成图像的图像缓冲区的大小')
        parser.add_argument('--lr_policy', type=str, default='linear', help='学习率策略。[linear|step|plateau|cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='每lr_decay_iters迭代乘以一个γ')

        self.isTrain = True
        return parser
