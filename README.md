# DISARM: Disentangled Scanner-free Image Generation via Unsupervised Image2Image Translation

MRI scanners can produce images with varying contrast, brightness, and spatial characteristics due to differences in hardware, software, calibration, and other factors such as machine maintenance, protocols, environmental conditions, and operator expertise. In multicenter studies involving diverse scanners and centers, this variability can confound results, underscoring the critical need for harmonized images to ensure robust analysis.
We propose a harmonization model for 3D medical images, with specific application to T1-weighted brain MRI data. Particularly, our model targets the scanner-free generation of 3D MR images to clean out clinical images from batch effects.

The three key objectives of the model are:

- Create a scanner-free space that enables the uniform transfer of images across different scanners in a denoised setting

- Impart the appearance of a specific training scanner to images, transferring its unique characteristics

- Avoid time-consuming preprocessing of MR images

## MRI Preprocessing

The preprocessing of MR images for the model includes the following steps:

1. Reorient the image to the standard orientation [(fslreorient2std)](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/utilities/fslutils)
2. Perform bias-field correction [(FAST)](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/fast)
3. Register the image to the reference Standard MNI152-T1-1mm [(FLIRT)](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/)

To automate these preprocessing steps, utilize the provided `MRI_Preprocessing.sh` script. 
Run the script using:

```
chmod +x MRI_Preprocessing.sh
./MRI_Preprocessing.sh /path/folder_with_raw_mri_images /path/new_folder_with_prep_images
```
The script will process all images in the specified `folder_with_raw_mri_images` and save the preprocessed images to `new_folder_with_prep_images`. The images must have extension `.nii.gz`.

## Dataset Setup

To set up the subfolders for training within `/path/dataset`, use the `create_scanner_folders()` function found in the `Load_Dataset.py` file. 
Detailed instructions on how to use this function are provided within the .py file itself.

## Training

Clone the repository:

```
git clone https://github.com/luca2245/DISARM_Harmonization.git
cd DISARM_Harmonization/Code
```

Start Training:
```
python main.py --dataroot /path/dataset --nThreads 8 --batch_size 2 --num_domains 5 --input_dim 1
--result_dir path/Result --display_dir path/Logs --d_iter 2 --n_ep 100000 --img_save_freq 500 
--model_save_freq 1000 --isDcontent
```

- The image slices in the three dimensions (sagittal, axial, and coronal) from all reconstructions performed by the model during training are saved to `--result_dir` at intervals defined by `--img_save_freq`. 
- The model is saved to `--result_dir` at intervals specified by `--model_save_freq`.
- Use `--num_domains` to select the number of training scanner.

## Test

Use the `Testing.py` file to load the trained model and a new image from any scanner (the scanner can be also different from those used in training).

- The `transfer_to_reference_scanner()` function converts an image, `source_img`, to the reference training scanner space specified by `ref_scanner` (e.g., 0 for the first training scanner). 

- The `transfer_to_scannerfree()` function converts an image, `source_img`, into the scanner-free space.

