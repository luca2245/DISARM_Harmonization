import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.optim import lr_scheduler
from torch.autograd import Variable
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
from Load_Dataset import *


def get_axis_slices(img, slice_index):
   num_rows = 1
   num_cols = 3
   names = ['Sagittal', 'Axial', 'Coronal']
   fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))

   for j in range(num_cols):
           if(j == 0):
               z_slice = img[:, :, slice_index]
           if(j == 1):
               z_slice = img[slice_index, :, :]
           if(j == 2):
               z_slice = img[:, (slice_index+18), :]
           # Plot the 2D slice in the current axis
           axes[j].imshow(z_slice, cmap='gray')
           axes[j].set_title(f'T1 - {names[j]}', fontsize=20)
   plt.tight_layout()
   plt.show()

def get_axis_slices_90_rot(img, slice_index):
   num_rows = 1
   num_cols = 3
   names = [ 'Sagittal','Coronal','Axial' ]
   fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))

   for j in range(num_cols):
           if(j == 0):
               z_slice = np.rot90(img[slice_index, :, :])
           if(j == 1):
               z_slice = np.rot90(img[:, (slice_index+18), :])
           if(j == 2):
               z_slice = np.rot90(img[:, :, slice_index])
           # Plot the 2D slice in the current axis
           axes[j].imshow(z_slice, cmap='gray')
           axes[j].set_title(f'T1 - {names[j]}', fontsize=20)
   plt.tight_layout()
   plt.show()

def tensor2img(img):
  img = img.cpu().float().detach().numpy()
  img = (img + 1) / 2.0 * 255.0
  return img.astype(np.uint8)

def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

# Training and Test options
class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--num_domains', type=int, default=3)
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--input_dim', type=int, default=1, help='# of input channels')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')

    # training related
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--lambda_rec', type=float, default=10)
    self.parser.add_argument('--lambda_cls', type=float, default=3.0)
    self.parser.add_argument('--lambda_cls_G', type=float, default=10.0)
    self.parser.add_argument('--isDcontent', action='store_true')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--num_domains', type=int, default=3)
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim', type=int, default=1, help='# of input channels for domain A')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./outputs', help='path for saving result images and models')

    # model related
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt


