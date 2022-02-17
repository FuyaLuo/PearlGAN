import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d
import torch.nn.functional as F
from pytorch_msssim import SSIM



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Spectral normalization base class, https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py 
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input
 

class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

################################SN#######################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

def define_G(input_nc, output_nc, ngf, net_Gen_type, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if net_Gen_type == 'gen_v1':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoderv1, enc_args, ResnetGenDecoderv1, dec_args)
    elif net_Gen_type == 'ori':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args)
    else:
        raise NotImplementedError('Generation Net [%s] is not found' % net_Gen_type)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG

def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminatorSN, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))
######Optional added by lfy

class Get_gradmag_gray(nn.Module):
    "To obtain the magnitude values of the gradients at each position."
    def __init__(self):
        super(Get_gradmag_gray, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x_norm = (x + 1) / 2
        x_norm = (.299*x_norm[:,0:1,:,:] + .587*x_norm[:,1:2,:,:] + .114*x_norm[:,2:3,:,:])
        x0_v = F.conv2d(x_norm, self.weight_v, padding = 1)
        x0_h = F.conv2d(x_norm, self.weight_h, padding = 1)

        x_gradmagn = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x_gradmagn

def GradMagIRLossv8(real_IR_edgemap, fake_vis_gradmap, sqrt_patch_num, gradient_th):
    "SGA Loss. The ratio of the gradient at the edge location to the maximum gradient in "
    "its neighborhood is encouraged to be greater than a given threshold."

    b, c, h, w = fake_vis_gradmap.size()
    # patch_num = sqrt_patch_num * sqrt_patch_num
    AAP_module = nn.AdaptiveAvgPool2d(sqrt_patch_num)
    real_IR_edgemap_pooling = AAP_module(real_IR_edgemap.expand_as(fake_vis_gradmap))
    if torch.sum(real_IR_edgemap) > 0:
        pooling_array = real_IR_edgemap_pooling[0].detach().cpu().numpy()
        h_nonzero, w_nonzero = np.nonzero(pooling_array[0])
        patch_idx_rand = np.random.randint(0, len(h_nonzero))
        patch_idx_x = h_nonzero[patch_idx_rand]
        patch_idx_y = w_nonzero[patch_idx_rand]
        crop_size = h // sqrt_patch_num
        pos_list = list(range(0, h, crop_size))

        pos_h = pos_list[patch_idx_x]
        pos_w = pos_list[patch_idx_y]
        # rand_patch = self.Tensor(b, c, crop_size, crop_size)
        rand_edgemap_patch = real_IR_edgemap[:, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
        rand_gradmap_patch = fake_vis_gradmap[:, :, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]

        sum_edge_pixels = torch.sum(rand_edgemap_patch) + 1
        # print('Sum_edge_pixels of IR edge map is: ', sum_edge_pixels.detach().cpu().numpy())
        fake_grad_norm = rand_gradmap_patch / torch.max(rand_gradmap_patch)
        losses = (torch.sum(F.relu(gradient_th * rand_edgemap_patch - fake_grad_norm))) / sum_edge_pixels
    else:
        losses = 0

    return losses

def DiversityAttLossv2(concat_att_maps):
    "The sum of the attention weights for each position is encouraged to be 1."

    losses = torch.mean(0.25 * ((torch.sum(concat_att_maps, dim=1)- 1.0) ** 2)) 
    return losses

def DiversityAttLossv3(concat_att_maps, input_fea, gpu_ids):
    "The similarity of features across scales is constrained to be as small as possible to "
    "achieve compactness of the attention distribution."

    avefea, _ = WeiAveAttFea(input_fea, concat_att_maps, 0, gpu_ids)
    avefea_matrix = avefea[0, 0, :, :]
    fea_cos_similarity = torch.mm(avefea_matrix, torch.transpose(avefea_matrix, 0, 1))
    _, c_a, _, _ = concat_att_maps.size()
    losses = F.relu(torch.mean(fea_cos_similarity - fea_cos_similarity.mul(torch.eye(c_a).cuda(gpu_ids))))

    return losses

def IntegratedAttLoss(concat_att_maps):
    "The maximum attention weight for each position is encouraged to be 1."

    att_max_weight = torch.max(concat_att_maps, 1)
    losses = 1.0 - torch.mean(att_max_weight[0])
    return losses

def WeiAveAttFea(input_tensor, att_maps, att_th, gpu_ids=[]):
    "Returns an L2-parametric normalized weighted attentional feature and an attentional weight."
    
    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(b, 1, c_a, c).cuda(gpu_ids)
    att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(c_a):
            temp_tensor = att_maps[0, i, :, :]
            if att_th == 0.0:
                mask_att = temp_tensor
            else:
                mask_HP = torch.zeros_like(temp_tensor).cuda(gpu_ids)
                mask_HP = torch.where(temp_tensor > att_th, torch.ones_like(mask_HP), mask_HP)
                mask_att = (mask_HP.expand_as(temp_tensor)).mul(temp_tensor)
            temp_tensor_max = torch.max(temp_tensor)
            # attmap_sm = (sm(temp_tensor.view(b, 1, -1))).view(b, 1, h, -1)
            att_fea_map = (mask_att.expand_as(input_tensor)).mul(input_tensor)
            ave_fea = (torch.squeeze(GAP(att_fea_map) * h * w)) / torch.sum(mask_att)    # b * c * 1 * 1
            out_tensor[0, 0, i, :] = ave_fea
            att_maps_max_value[0, 0, i, 0] = temp_tensor_max
    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor_L2norm, att_maps_max_value

def AttAlignLossv2(A_feature, A_attmaps, B_feature, B_attmaps, gpu_ids=[]):
    "Cross-domain images are encouraged to have greater similarity of attentional features "
    "at the same scale than across scales."

    A_avefea, A_att_weight = WeiAveAttFea(A_feature, A_attmaps, 0, gpu_ids)
    B_avefea, B_att_weight = WeiAveAttFea(B_feature, B_attmaps, 0, gpu_ids)
    A_avefea_matrix = A_avefea[0, 0, :, :]
    B_avefea_matrix = B_avefea[0, 0, :, :]
    fea_cos_similarity = torch.mm(A_avefea_matrix, torch.transpose(B_avefea_matrix, 0, 1))
    fea_cossim_max = torch.max(fea_cos_similarity, dim=1)
    corr_fea_cossim = torch.zeros_like(fea_cossim_max[0]).cuda(gpu_ids)
    _, c_a, _, _ = A_attmaps.size()
    for i in range(c_a):
        corr_fea_cossim[i] = fea_cos_similarity[i, i]

    att_weight_matrix = torch.zeros((c_a, 2)).cuda(gpu_ids)
    att_weight_matrix[:, 0] = A_att_weight[0, 0, :, 0]
    att_weight_matrix[:, 1] = B_att_weight[0, 0, :, 0]
    att_loss_weight = torch.min(att_weight_matrix, dim=1)[0] #The min for max value of attention map at the same scale.
    # print(att_weight_matrix)

    cossim_diff = fea_cossim_max[0] - corr_fea_cossim
    # print(cossim_diff)
    loss_align = (att_loss_weight[0] * cossim_diff[0] + att_loss_weight[1] * cossim_diff[1] + \
                att_loss_weight[2] * cossim_diff[2]) / (att_loss_weight[0] + att_loss_weight[1] + att_loss_weight[2] + 1e-6)

    loss_diversity = DiversityAttLossv3(A_attmaps, A_feature, gpu_ids) + DiversityAttLossv3(B_attmaps, B_feature, gpu_ids)
    losses = loss_align + 0.25 * loss_diversity
    # print(loss_align)

    return losses

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return ( 1 - super(SSIM_Loss, self).forward(img1, img2) )

# Defines the total variation (TV) loss, which encourages spatial smoothness in the translated image.
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

###Add 1 TDGA Block before ResBlock groups.
class ResnetGenEncoderv1(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoderv1, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        model += [PGAResBlockv4k3(ngf * mult, norm_layer=norm_layer, use_bias=use_bias)]
        model_res = []
        for _ in range(n_blocks):
            model_res += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)
        self.model_postfix = nn.Sequential(*model_res)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            out1, attmap1, attmap2, attmap3, attmap4 = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            out2 = nn.parallel.data_parallel(self.model_postfix, out1, self.gpu_ids)
        else:
            out1, attmap1, attmap2, attmap3, attmap4 = self.model(input)
            out2 = self.model_postfix(out1)
        return out2, attmap1, attmap2, attmap3, attmap4


####Add 1 TDGA Block before ResBlock groups.
class ResnetGenDecoderv1(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoderv1, self).__init__()
        self.gpu_ids = gpu_ids
        
        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
        self.model_att = PGAResBlockv4k3(ngf * mult, norm_layer=norm_layer, use_bias=use_bias)

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      nn.GroupNorm(32, int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            out1, _, _, _, _ = nn.parallel.data_parallel(self.model_att, input, self.gpu_ids)
            out2 = nn.parallel.data_parallel(self.model, out1, self.gpu_ids)
        else:
            out1, _, _, _, _ = self.model_att(input)
            out2 = self.model(out1)

        return out2


class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

# Define a top-down guided attention module.
class PGAResBlockv4k3(nn.Module):
    def __init__(self, in_dim, norm_layer, use_bias):
        super(PGAResBlockv4k3, self).__init__()

        self.width = in_dim // 4
        self.bottlenec1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())

        self.ds1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.ds2 = nn.AvgPool2d(kernel_size=3, stride=4)
        self.ds3 = nn.AvgPool2d(kernel_size=3, stride=8)
        self.ds4 = nn.AvgPool2d(kernel_size=3, stride=16)

        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=use_bias), norm_layer(in_dim), nn.PReLU())

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, input):
        b, c, h, w = input.size()

        input_fea = self.conv(input)
        spx = torch.split(input_fea, self.width, 1)
        fea_ds1 = self.ds1(spx[0])
        fea_ds2 = self.ds2(spx[1])
        fea_ds3 = self.ds3(spx[2])
        fea_ds4 = self.ds4(spx[3])

        att_conv1 = self.bottlenec1(fea_ds4)
        att_map1_us = F.interpolate(att_conv1, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g1 = F.interpolate(att_conv1, size=(fea_ds3.size(2), fea_ds3.size(3)), mode='bilinear', align_corners=False)
        
        fea_att1 = att_map_g1.expand_as(fea_ds3).mul(fea_ds3) + fea_ds3
        att_conv2 = self.bottlenec2(fea_att1)
        att_map2_us = F.interpolate(att_conv2, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g2 = F.interpolate(att_conv2, size=(fea_ds2.size(2), fea_ds2.size(3)), mode='bilinear', align_corners=False)

        fea_att2 = att_map_g2.expand_as(fea_ds2).mul(fea_ds2) + fea_ds2
        att_conv3 = self.bottlenec3(fea_att2)
        att_map3_us = F.interpolate(att_conv3, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g3 = F.interpolate(att_conv3, size=(fea_ds1.size(2), fea_ds1.size(3)), mode='bilinear', align_corners=False)
        
        fea_att3 = att_map_g3.expand_as(fea_ds1).mul(fea_ds1) + fea_ds1
        att_conv4 = self.bottlenec4(fea_att3)
        att_map4_us = F.interpolate(att_conv4, size=(h, w), mode='bilinear', align_corners=False)
        
        y1 = att_map4_us.expand_as(spx[0]).mul(spx[0])
        y2 = att_map3_us.expand_as(spx[1]).mul(spx[1])
        y3 = att_map2_us.expand_as(spx[2]).mul(spx[2])
        y4 = att_map1_us.expand_as(spx[3]).mul(spx[3])

        out = torch.cat((y1, y2, y3, y4), 1) + input

        return out, att_map1_us, att_map2_us, att_map3_us, att_map4_us

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
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
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3

class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminatorSN, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            SNConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                SNConv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            SNConv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.PReLU(),
            \
            SNConv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3


class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            # net = nn.DataParallel(net).cuda(device_id) #edited by lfy
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def forward(self, input, in_domain, out_domain):
        encoded, attmap1, attmap2, attmap3, attmap4 = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output, attmap1, attmap2, attmap3, attmap4 = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain), attmap1, attmap2, attmap3, attmap4
        return output, attmap1, attmap2, attmap3, attmap4

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
