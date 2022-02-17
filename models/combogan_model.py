import numpy as np
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ComboGANModel(BaseModel):
    def name(self):
        return 'ComboGANModel'

    def __init__(self, opt):
        super(ComboGANModel, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.net_Gen_type,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, self.gpu_ids)
            self.EdgeMap_A = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
            self.EdgeMap_B = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        #######################################
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.L1 = torch.nn.SmoothL1Loss()
            self.downsample = torch.nn.AvgPool2d(3, stride=2)
            self.criterionCycle = self.L1
            self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t)) 
            self.criterionLatent = lambda y,t : self.L1(y, t.detach())
            self.criterionGAN = lambda r,f,v : (networks.GANLoss(r[0],f[0],v) + \
                                                networks.GANLoss(r[1],f[1],v) + \
                                                networks.GANLoss(r[2],f[2],v)) / 3  
            self.criterionGradMagIR = lambda y,t,r,v : networks.GradMagIRLossv8(y, t, r, v)
            self.criterionGradMagVis = lambda y,t,r,v : networks.GradMagIRLossv8(y, t, r, v)
            self.criterionDivAtt = lambda y : networks.DiversityAttLossv2(y)
            self.criterionIntAtt = lambda y : networks.IntegratedAttLoss(y)
            self.criterionAttAlign = lambda y,t,r,v,g : networks.AttAlignLossv2(y, t, r, v, g)
            self.criterionSSIM = networks.SSIM_Loss(win_size=self.opt.ssim_winsize, data_range=1.0, size_average=True, channel=3)
            self.criterionTV = networks.TVLoss(TVLoss_weight=1)
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle = [0]*self.n_domains
            self.loss_sga, self.loss_tv, self.loss_ad = [0]*self.n_domains, [0]*self.n_domains, [0]*self.n_domains
            self.loss_accs = [0]*self.n_domains

            # initialize loss multipliers
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward
            self.get_gradmag = networks.Get_gradmag_gray()
            self.patch_num_sqrt = opt.sqrt_patch_num
            self.grad_th_vis = opt.grad_th_vis
            self.grad_th_IR = opt.grad_th_IR
            self.lambda_ad = opt.lambda_ad
            self.lambda_accs = opt.lambda_accs

            # self.lambda_ssim = opt.lambda_ssim
            self.SGA_start_ep = opt.SGA_start_epoch
            self.SGA_fullload_ep = opt.SGA_fullload_epoch
            if self.SGA_fullload_ep == 0:
                self.lambda_sga = opt.lambda_sga
                self.lambda_tv = opt.lambda_tv
            else:
                self.lambda_sga = 0.0
                self.lambda_tv = 0.0

            self.SSIM_start_ep = opt.SSIM_start_epoch
            self.SSIM_fullload_ep = opt.SSIM_fullload_epoch
            if self.SSIM_fullload_ep == 0:
                self.lambda_ssim = opt.lambda_ssim
            else:
                self.lambda_ssim = 0.0

            self.ACCS_start_ep = opt.ACCS_start_epoch
            self.loss_accs_fea_idx = 0.0
            if self.ACCS_start_ep == 0:
                self.loss_accs_idx = 1.0
            else:
                self.loss_accs_idx = 0.0

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
            input_EM_A = input['EMA']
            self.EdgeMap_A.resize_(input_EM_A.size()).copy_(input_EM_A)
            input_EM_B = input['EMB']
            self.EdgeMap_B.resize_(input_EM_B.size()).copy_(input_EM_B)
        self.image_paths = input['path']
#######################################

    def test(self, output_only=False):
        with torch.no_grad():
            if output_only:
                # cache encoding to not repeat it everytime
                encoded, _, _, _, _ = self.netG.encode(self.real_A, self.DA)
                for d in range(self.n_domains):
                    if d == self.DA and not self.opt.autoencode:
                        continue
                    fake = self.netG.decode(encoded, d)
                    self.visuals = [fake]
                    self.labels = ['fake_%d' % d]
            else:
                self.visuals = [self.real_A]
                self.labels = ['real_%d' % self.DA]

                # cache encoding to not repeat it everytime
                encoded, _, _, _, _ = self.netG.encode(self.real_A, self.DA)
                for d in range(self.n_domains):
                    if d == self.DA and not self.opt.autoencode:
                        continue
                    fake = self.netG.decode(encoded, d)
                    self.visuals.append( fake )
                    self.labels.append( 'fake_%d' % d )
                    if self.opt.reconstruct:
                        rec = self.netG.forward(fake, d, self.DA)
                        self.visuals.append( rec )
                        self.labels.append( 'rec_%d' % d )

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, pred_real, fake, domain):
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D = self.criterionGAN(pred_real, pred_fake, True) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.pred_real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.pred_real_A, fake_A, self.DA)

    def backward_G(self):
        encoded_A, self.A_attmap1_enc, self.A_attmap2_enc, self.A_attmap3_enc, self.A_attmap4_enc = self.netG.encode(self.real_A, self.DA)
        encoded_B, self.B_attmap1_enc, self.B_attmap2_enc, self.B_attmap3_enc, self.B_attmap4_enc = self.netG.encode(self.real_B, self.DB)

        # Optional identity "autoencode" loss
        if self.lambda_idt > 0:
            # Same encoder and decoder should recreate image
            idt_A = self.netG.decode(encoded_A, self.DA)
            loss_idt_A = self.criterionIdt(idt_A, self.real_A)
            idt_B = self.netG.decode(encoded_B, self.DB)
            loss_idt_B = self.criterionIdt(idt_B, self.real_B)
        else:
            loss_idt_A, loss_idt_B = 0, 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.decode(encoded_A, self.DB)
        pred_fake = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(self.pred_real_B, pred_fake, False)
        # D_B(G_B(B))
        self.fake_A = self.netG.decode(encoded_B, self.DA)
        pred_fake = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(self.pred_real_A, pred_fake, False)
        # Forward cycle loss
        rec_encoded_A, self.A_attmap1_rec, self.A_attmap2_rec, self.A_attmap3_rec, self.A_attmap4_rec = self.netG.encode(self.fake_B, self.DB)
        self.rec_A = self.netG.decode(rec_encoded_A, self.DA)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_cyc + \
                                    (self.criterionSSIM((self.rec_A + 1) / 2, (self.real_A + 1) / 2)) * self.lambda_ssim
        # Backward cycle loss
        rec_encoded_B, self.B_attmap1_rec, self.B_attmap2_rec, self.B_attmap3_rec, self.B_attmap4_rec = self.netG.encode(self.fake_A, self.DA)
        self.rec_B = self.netG.decode(rec_encoded_B, self.DB)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_cyc + \
                                    (self.criterionSSIM((self.rec_B + 1) / 2, (self.real_B + 1) / 2)) * self.lambda_ssim

        # Optional structured gradient alignment loss
        self.loss_sga[self.DA] = self.lambda_sga * self.criterionGradMagVis(self.EdgeMap_A, self.get_gradmag(self.fake_B), self.patch_num_sqrt, self.grad_th_vis)
        self.loss_sga[self.DB] = self.lambda_sga * self.criterionGradMagIR(self.EdgeMap_B, self.get_gradmag(self.fake_A), self.patch_num_sqrt, self.grad_th_IR)

        # Optional total variation loss
        self.loss_tv[self.DA] = self.lambda_tv * self.criterionTV(self.fake_A)
        self.loss_tv[self.DB] = self.lambda_tv * self.criterionTV(self.fake_B)

        # Optional attentional loss
        A_attmap_enc_concat = torch.cat((self.A_attmap2_enc, self.A_attmap3_enc, self.A_attmap4_enc), 1)
        B_attmap_enc_concat = torch.cat((self.B_attmap2_enc, self.B_attmap3_enc, self.B_attmap4_enc), 1)
        # Attentional diversity loss
        self.loss_ad[self.DA] = self.lambda_ad * 0.5 * (self.criterionDivAtt(A_attmap_enc_concat) + self.criterionIntAtt(A_attmap_enc_concat))
        self.loss_ad[self.DB] = self.lambda_ad * 0.5 * (self.criterionDivAtt(B_attmap_enc_concat) + self.criterionIntAtt(B_attmap_enc_concat))
        # Attentional cross-domain conditional similarity loss
        if self.loss_accs_idx == 0.0:
            self.loss_accs[self.DA] = 0.0
            self.loss_accs[self.DB] = 0.0
        else:
            self.loss_accs[self.DA] = self.lambda_accs * self.criterionAttAlign(encoded_A.detach(), A_attmap_enc_concat, rec_encoded_B.detach(), B_attmap_enc_concat, self.gpu_ids[0])
            self.loss_accs[self.DB] = self.lambda_accs * self.criterionAttAlign(encoded_B.detach(), B_attmap_enc_concat, rec_encoded_A.detach(), A_attmap_enc_concat, self.gpu_ids[0])
            
        ######################################

        # Optional cycle loss on encoding space
        if self.lambda_enc > 0:
            loss_enc_A = self.criterionLatent(rec_encoded_A, encoded_A)
            loss_enc_B = self.criterionLatent(rec_encoded_B, encoded_B)
        else:
            loss_enc_A, loss_enc_B = 0, 0

        # Optional loss on downsampled image before and after
        if self.lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.fake_B, self.real_A)
            loss_fwd_B = self.criterionIdt(self.fake_A, self.real_B)
        else:
            loss_fwd_A, loss_fwd_B = 0, 0


        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) + \
                 (loss_idt_A + loss_idt_B) * self.lambda_idt + \
                 (loss_enc_A + loss_enc_B) * self.lambda_enc + \
                 (loss_fwd_A + loss_fwd_B) * self.lambda_fwd + (self.loss_sga[self.DA] + self.loss_sga[self.DB]) + \
                 (self.loss_tv[self.DA] + self.loss_tv[self.DB]) + (self.loss_ad[self.DA] + self.loss_ad[self.DB]) + \
                 (self.loss_accs[self.DA] + self.loss_accs[self.DB])

                
        loss_G.backward()

    def optimize_parameters(self):

        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)
        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)
        
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)
        

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]

        D_losses, G_losses, cyc_losses, attali_losses, att_losses, sga_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle), extract(self.loss_accs), extract(self.loss_ad), extract(self.loss_sga)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses), ('AttAli', attali_losses), ('ATT', att_losses), ('SGA', sga_losses)])   

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def get_current_visuals2(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
            self.attmaps = [self.A_attmap1_enc, self.A_attmap2_enc, self.A_attmap3_enc, self.A_attmap4_enc, self.B_attmap1_enc, self.B_attmap2_enc, self.B_attmap3_enc, self.B_attmap4_enc]
            self.attmaps_input = [self.real_A, self.real_A, self.real_A, self.real_A, self.real_B, self.real_B, self.real_B, self.real_B]
            self.map_labels = ['A_map1', 'A_map2', 'A_map3', 'A_map4', 'B_map1', 'B_map2', 'B_map3', 'B_map4']
            images = [util.tensor2im(v.data) for v in self.visuals]
            # print(type(images))
            Att_maps = [util.visual_attmaps(u.data, v.data) for u,v in zip(self.attmaps_input, self.attmaps)]
            out = OrderedDict(zip(self.labels + self.map_labels, images + Att_maps))
        else:
            images = [util.tensor2im(v.data) for v in self.visuals]
            out = OrderedDict(zip(self.labels, images))
        return out

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_enc = self.opt.lambda_latent * decay_frac

        if self.SSIM_fullload_ep != 0:
            if curr_iter > (self.SSIM_start_ep - 1):
                if curr_iter > (self.SSIM_fullload_ep - 1):
                    self.lambda_ssim = self.opt.lambda_ssim


        if self.ACCS_start_ep != 0:
            if curr_iter > (self.ACCS_start_ep - 1):
                self.loss_accs_idx = 1.0
            
