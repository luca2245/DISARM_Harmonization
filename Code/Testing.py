from Utils import *
from Load_Dataset import *
from DISARM_class import *
import nibabel as nib

def transfer_to_reference_scanner(source_img, ref_scanner, opts, model):
        dataset_dom = data_single_std(opts, ref_scanner)
        dataloader_dom = torch.utils.data.DataLoader(dataset_dom, batch_size=1, shuffle=True, num_workers=6)
        for batch in dataloader_dom:
            img, lab = batch
        img_ = img.cuda().float()
        lab_ = lab.cuda().float()
        source_img_ = source_img.cuda().float()
        with torch.no_grad():
            output_test = model.test_reference_transfer(image = source_img_, image_trg = img_, c_trg = lab_)
        return output_test[0,0]

def get_z_random(batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz)
    return z

def transfer_to_scannerfree(source_img, opts, model):
        z_random = get_z_random(source_img.size(0), 8, 'gauss').cuda().float()
        source_img_ = source_img.cuda().float()
        with torch.no_grad():
            output_test = model.test_scannerfree_transfer(source_img_, z_random)
        return output_test[0,0]

args = [
    '--dataroot', 'path/test_images',  # Specify the path to your dataset
    '--nThreads', '8',
    '--resume', 'path/last.pth',
    '--result_dir', 'path/result_dir',
    '--phase', 'train',
    '--input_dim', '1',
    '--num_domains', '5',
    '--gpu', '0'
]

options = TestOptions()
opts = options.parser.parse_args(args)
opts.dis_scale = 3
opts.dis_norm = 'None'
opts.dis_spectral_norm = False

# LOAD MODEL 
print('\n--- load model ---')
model = DISARM(opts)
model.setgpu(opts.gpu)
model.resume(opts.resume, train=False)
model.eval()

# Load a new image
transforms = [ToTensor()]
transforms.append(lambda x: x.unsqueeze(0))
transforms.append(lambda x: x.permute(0, 2, 3 , 1))
transforms.append(tio.RescaleIntensity((-1, 1)))
transforms = Compose(transforms)

new_img = nib.load(f'/path/new_img.nii.gz')
new_img = new_img.get_fdata()
new_img = transforms(new_img)
new_img = new_img.unsqueeze(0)

get_axis_slices_90_rot(new_img[0,0], 85)

# Transfer using as reference a training scanner
result = transfer_to_reference_scanner(new_img, 0, opts, model)

# Transfer within the scanner-free space
result = transfer_to_scannerfree(new_img, opts, model)


get_axis_slices_90_rot(tensor2img(result), 85)


