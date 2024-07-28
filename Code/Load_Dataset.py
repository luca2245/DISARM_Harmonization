import os
import sys
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor
import random
import numpy as np
import torch
import torchio as tio
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# The function creates, inside **data_folder_path**, subfolders named TrainA, TrainB, TrainC, etc., corresponding to the number 
# of training scanners. The **image_arrays** parameter must be a list where each element is itself a list of images from a specific 
# scanner. The function will then place the images from the first element of **image_arrays** into the TrainA subfolder, 
# the images from the second element into the TrainB subfolder, and so on.
def create_scanner_folders(data_folder_path, num_scanners, image_arrays):
    base_folder = data_folder_path
    domain_list = [f"train{chr(65 + i)}" for i in range(num_scanners)]

    # Create folders
    for domain in domain_list:
        folder_path = os.path.join(base_folder, domain)
        os.makedirs(folder_path, exist_ok=True)
        
    for i, domain_images in enumerate(image_arrays):
        domain = domain_list[i]
        folder_path = os.path.join(base_folder, domain)

        for j, image_array in enumerate(domain_images):
            # Construct the destination path
            array_name = f'image{j + 1}.npy'  # Modify the naming convention and extension if needed
            destination_path = os.path.join(folder_path, array_name)

            # Save the numpy array as .npy file
            np.save(destination_path, image_array)



def custom_transform(x):
    return x.unsqueeze(0)

def custom_permute(x):
    return x.permute(0, 2, 3, 1)


# Standard Loader
class data_multi_std(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.num_domains = opts.num_domains
        self.input_dim = opts.input_dim

        domains = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.images = [None] * self.num_domains
        stats = ''

        for i in range(self.num_domains):
            img_dir = os.path.join(self.dataroot, opts.phase + domains[i])
            ilist = os.listdir(img_dir)
            self.images[i] = [os.path.join(img_dir, x) for x in ilist]
            stats += '{}: {}'.format(domains[i], len(self.images[i]))

        stats += ' images'
        self.dataset_size = max([len(self.images[i]) for i in range(self.num_domains)])

        # Image Transformation
        transforms = [ToTensor()]
        transforms.append(custom_transform)
        transforms.append(custom_permute)
        transforms.append(tio.RescaleIntensity((-1, 1)))

        self.transforms = Compose(transforms)

    def __getitem__(self, index):
        cls = random.randint(0, self.num_domains - 1)
        c_org = np.zeros((self.num_domains,))
        data = self.load_img(self.images[cls][random.randint(0, len(self.images[cls]) - 1)], self.input_dim)
        c_org[cls] = 1
        #data = data.permute(0, 1, 3, 2)
        return data, torch.FloatTensor(c_org)

    def load_img(self, img_name, input_dim):
        img = np.load(img_name)
        img = self.transforms(img)
        return img

    def __len__(self):
        return self.dataset_size


class data_single_std(data.Dataset):
  def __init__(self, opts, domain):
    self.dataroot = opts.dataroot
    domains = [chr(i) for i in range(ord('A'),ord('Z')+1)]
    images = os.listdir(os.path.join(self.dataroot, opts.phase + domains[domain]))
    self.img = [os.path.join(self.dataroot, opts.phase + domains[domain], x) for x in images]
    self.size = len(self.img)
    self.input_dim = opts.input_dim

    self.c_org = np.zeros((opts.num_domains,))
    self.c_org[domain] = 1
    # Image Transformation
    transforms = [ToTensor()]
    transforms.append(lambda x: x.unsqueeze(0))
    transforms.append(lambda x: x.permute(0, 2, 3 , 1))
    transforms.append(tio.RescaleIntensity((-1, 1)))

    self.transforms = Compose(transforms)
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = np.load(img_name)
    img = self.transforms(img)
    return img, self.c_org

  def __len__(self):
    return self.size


# ### Loader with Augmentation
# We utilize the **tio.RandomElasticDeformation** command for augmentation.
# 
# It applies a random elastic deformation to the anatomical structure of the image.

transforms_dict = {
    tio.RandomElasticDeformation(num_control_points=10,  # or just 7
    locked_borders=2, max_displacement = 8): 0.7,
    tio.Lambda(lambda x: x): 0.3
}

class data_multi_aug(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.num_domains = opts.num_domains
        self.input_dim = opts.input_dim

        domains = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.images = [None] * self.num_domains
        stats = ''

        for i in range(self.num_domains):
            img_dir = os.path.join(self.dataroot, opts.phase + domains[i])
            ilist = os.listdir(img_dir)
            self.images[i] = [os.path.join(img_dir, x) for x in ilist]
            stats += '{}: {}'.format(domains[i], len(self.images[i]))

        stats += ' images'
        self.dataset_size = max([len(self.images[i]) for i in range(self.num_domains)])

        # Image Transformation and Augmentation
        transforms = [ToTensor()]
        transforms.append(custom_transform)
        transforms.append(custom_permute)
        transforms.append(tio.RescaleIntensity((-1, 1)))
        transforms.append( tio.OneOf(transforms_dict) )

        self.transforms = Compose(transforms)

    def __getitem__(self, index):
        cls = random.randint(0, self.num_domains - 1)
        c_org = np.zeros((self.num_domains,))
        data = self.load_img(self.images[cls][random.randint(0, len(self.images[cls]) - 1)], self.input_dim)
        c_org[cls] = 1
        #data = data.permute(0, 1, 3, 2)
        return data, torch.FloatTensor(c_org)

    def load_img(self, img_name, input_dim):
        img = np.load(img_name)
        img = self.transforms(img)
        return img

    def __len__(self):
        return self.dataset_size


class data_single_aug(data.Dataset):
  def __init__(self, opts, domain):
    self.dataroot = opts.dataroot
    domains = [chr(i) for i in range(ord('A'),ord('Z')+1)]
    images = os.listdir(os.path.join(self.dataroot, opts.phase + domains[domain]))
    self.img = [os.path.join(self.dataroot, opts.phase + domains[domain], x) for x in images]
    self.size = len(self.img)
    self.input_dim = opts.input_dim

    self.c_org = np.zeros((opts.num_domains,))
    self.c_org[domain] = 1
    
    # Image Transformation and Augmentation
    transforms = [ToTensor()]
    transforms.append(lambda x: x.unsqueeze(0))
    transforms.append(lambda x: x.permute(0, 2, 3 , 1))
    transforms.append(tio.RescaleIntensity((-1, 1)))
    transforms.append( tio.OneOf(transforms_dict) )

    self.transforms = Compose(transforms)
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = np.load(img_name)
    img = self.transforms(img)
    return img, self.c_org

  def __len__(self):
    return self.size


