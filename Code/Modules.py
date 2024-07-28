from Basic_Blocks import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
import torch.nn.functional as F
from Load_Dataset import *
from Utils import *

# ENCODERS

### Anatomy Encoder
class Anatomy_Encoder(nn.Module):
  def __init__(self, input_dim):
    super(Anatomy_Encoder, self).__init__()
    enc_c = []
    tch = 4
    enc_c += [LeakyReLUConv3d(input_dim, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1, 2):
      enc_c += [ReLUINSConv3d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      enc_c += [INSResBlock3d(tch, tch)]

    for i in range(0, 1):
      enc_c += [INSResBlock3d(tch, tch)]
      enc_c += [GaussianNoiseLayer()]
    self.conv = nn.Sequential(*enc_c)

  def forward(self, x):
    return self.conv(x)

### Scanner_Encoder
class Scanner_Encoder(nn.Module):
  def __init__(self, input_dim, output_nc=16, c_dim=3, norm_layer=None, nl_layer=None):
    super(Scanner_Encoder, self).__init__()

    ndf = 8
    n_blocks=4
    max_ndf = 4

    conv_layers = [nn.ReflectionPad3d(1)]
    conv_layers += [nn.Conv3d(input_dim+c_dim, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  
      output_ndf = ndf * min(max_ndf, n+1)  
      conv_layers += [BasicBlock3d(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers += [nl_layer(), nn.AdaptiveAvgPool3d(1)] 
    self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv = nn.Sequential(*conv_layers)

  def forward(self, x, c):
    c = c.view(c.size(0), c.size(1), 1, 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3), x.size(4))
    x_c = torch.cat([x, c], dim=1)
    x_conv = self.conv(x_c)
    conv_flat = x_conv.view(x.size(0), -1)
    output = self.fc(conv_flat)
    outputVar = self.fcVar(conv_flat)
    return output, outputVar

# GENERATOR
class Generator(nn.Module):
  def __init__(self, output_dim, c_dim=3, nz=16):
    super(Generator, self).__init__()
    self.nz = nz
    self.c_dim = c_dim
    tch = 8
    dec_share = []
    dec_share += [INSResBlock3d(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 8+self.nz+self.c_dim
    dec1 = []
    for i in range(0, 3):
      dec1 += [INSResBlock3d(tch, tch)]
    tch = tch + self.nz
    dec2 = [ReLUINSConvTranspose3d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=0)]
    tch = tch//2
    tch = tch + self.nz
    dec3 = [ReLUINSConvTranspose3d(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
    tch = tch//2
    tch = tch + self.nz
    dec4 = [nn.ConvTranspose3d(tch, output_dim, kernel_size=2, stride=1, padding=0)]+[nn.Tanh()]
    self.dec1 = nn.Sequential(*dec1)
    self.dec2 = nn.Sequential(*dec2)
    self.dec3 = nn.Sequential(*dec3)
    self.dec4 = nn.Sequential(*dec4)

  def forward(self, x, z, c):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
        c = c.view(c.size(0), c.size(1), 1, 1, 1)
        c = c.repeat(1, 1, out0.size(2), out0.size(3), out0.size(4))
        x_c_z = torch.cat([out0, c, z_img], 1)
        out1 = self.dec1(x_c_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3), out1.size(4))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.dec2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3), out2.size(4))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.dec3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3), out3.size(4))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.dec4(x_and_z4)
        return out4


# DISCRIMINATORS

### Anatomy_Discriminator
class Anatomy_Discriminator(nn.Module):
  def __init__(self, c_dim=3):
    super(Anatomy_Discriminator, self).__init__()
    model = []
    model += [LeakyReLUConv3d(8, 6, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv3d(6, 6, kernel_size=5, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv3d(6, 6, kernel_size=3, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv3d(6, 6, kernel_size=3, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv3d(6, 8, kernel_size=2, stride=2, padding=0)]
    model += [nn.Conv3d(8, c_dim, kernel_size=2, stride=2, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    #out = out.view(-1)
    #out = out.view(out.size(0), out.size(1), out.size(2), out.size(3))
    out = out.view(out.size(0), out.size(1))

    return out

### Scanner_Discriminator
class Scanner_Discriminator(nn.Module):
  def __init__(self, input_dim, norm='None', sn=False, c_dim=3, image_size=216):
    super(Scanner_Discriminator, self).__init__()
    ch = 8
    n_layer = 6
    self.model, curr_dim = self._make_net(ch, input_dim, n_layer, norm, sn)
    self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=1, stride=1, padding=1, bias=False)
    kernal_size = int(image_size/np.power(2, n_layer))
    self.conv2 = nn.Conv3d(curr_dim, c_dim, kernel_size=kernal_size, bias=False)
    self.pool = nn.AdaptiveAvgPool3d(1)

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv3d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] 
    tch = ch
    for i in range(1, n_layer-1):
      model += [LeakyReLUConv3d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] 
      tch *= 2
    model += [LeakyReLUConv3d(tch, tch, kernel_size=2, stride=2, padding=1, norm='None', sn=sn)]
    return nn.Sequential(*model), tch

  def cuda(self,gpu):
    self.model.cuda(gpu)
    self.conv1.cuda(gpu)
    self.conv2.cuda(gpu)

  def forward(self, x):
    h = self.model(x)
    out = self.conv1(h)
    out_cls = self.conv2(h)
    out_cls = self.pool(out_cls)
    return out, out_cls.view(out_cls.size(0), out_cls.size(1))


