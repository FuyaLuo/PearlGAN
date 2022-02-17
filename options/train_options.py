from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load if continuing training')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc (determines name of folder to load from)')
        self.parser.add_argument('--IR_edge_path', type=str, default='./IR_edge_map/', help='the folder to load IR image edge map')
        self.parser.add_argument('--Vis_edge_path', type=str, default='./Vis_edge_map/', help='the folder to load Visible image edge map')
        self.parser.add_argument('--ssim_winsize', type=int, default=3, help='window size of SSIM loss')
        self.parser.add_argument('--sqrt_patch_num', type=int, default=8, help='the parts number for each side')
        self.parser.add_argument('--grad_th_vis', type=float, default=0.8, help='threshold for SGA Loss about gradient of fake IR image')
        self.parser.add_argument('--grad_th_IR', type=float, default=0.8, help='threshold for SGA Loss about gradient of fake Visible image')

        self.parser.add_argument('--niter', required=True, type=int, help='# of epochs at starting learning rate (try 50*n_domains)')
        self.parser.add_argument('--niter_decay', required=True, type=int, help='# of epochs to linearly decay learning rate to zero (try 50*n_domains)')

        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for ADAM')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')

        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.0, help='weight for identity "autoencode" mapping (A -> A)')
        self.parser.add_argument('--lambda_latent', type=float, default=0.0, help='weight for latent-space loss (A -> z -> B -> z)')
        self.parser.add_argument('--lambda_forward', type=float, default=0.0, help='weight for forward loss (A -> B; try 0.2)')
        self.parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight for SSIM loss')
        self.parser.add_argument('--lambda_tv', type=float, default=5.0, help='weight for TV loss')
        self.parser.add_argument('--lambda_ad', type=float, default=1.0, help='weight for AD loss')
        self.parser.add_argument('--lambda_accs', type=float, default=1.0, help='weight for ACCS loss')
        
        self.parser.add_argument('--lambda_sga', type=float, default=0.5, help='weight for structured gradient alignment loss')
        self.parser.add_argument('--SGA_start_epoch', type=int, default=0, help='# of epochs at starting structured gradient alignment loss')
        self.parser.add_argument('--SGA_fullload_epoch', type=int, default=0, help='# of epochs at starting structured gradient alignment loss with full loaded weights')
        self.parser.add_argument('--SSIM_start_epoch', type=int, default=10, help='# of epochs at starting SSIM loss')
        self.parser.add_argument('--SSIM_fullload_epoch', type=int, default=10, help='# of epochs at starting SSIM loss with full loaded weights')
        self.parser.add_argument('--ACCS_start_epoch', type=int, default=10, help='# of epochs at starting ACCS loss')
    
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
