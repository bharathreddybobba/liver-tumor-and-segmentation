# import all required libraries
import glob
import os
import random
import types

import cv2
import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from fastai.basics import *
from fastai.data.transforms import *
from fastai.vision.all import *
from ipywidgets import *
from matplotlib.pyplot import figure
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm


# Function to read NIfTI file
def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

# Load file paths into a DataFrame
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append((dirname, filename))

df_files = pd.DataFrame(files, columns=['dirname', 'filename'])
df_files = df_files.sort_values(by='filename')

# Map CT scan and label
df_files["mask_dirname"] = ""
df_files["mask_filename"] = ""

for i in range(131):
    ct = f"volume-{i}.nii"
    mask = f"segmentation-{i}.nii"

    df_files.loc[df_files['filename'] == ct, 'mask_filename'] = mask
    df_files.loc[df_files['filename'] == ct, 'mask_dirname'] = "/kaggle/input/liver-tumor-segmentation/segmentations"

df_files = df_files[df_files.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True)

# Read and preprocess NIfTI file
sample = 40
sample_ct = read_nii(df_files.loc[sample, 'dirname']+"/"+df_files.loc[sample, 'filename'])
sample_mask = read_nii(df_files.loc[sample, 'mask_dirname']+"/"+df_files.loc[sample, 'mask_filename'])

# Print CT and mask shapes
print(f'CT Shape:   {sample_ct.shape}\nMask Shape: {sample_mask.shape}')

# Print min and max values of CT and mask arrays
print(np.amin(sample_ct), np.amax(sample_ct))
print(np.amin(sample_mask), np.amax(sample_mask))

# Preprocessing
dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(150, 30),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60)
)

@patch
def windowed(self:Tensor, w, l):
    px = self.clone()
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)

# Plotting CT scan
figure(figsize=(8, 6), dpi=100)
plt.imshow(tensor(sample_ct[..., 55].astype(np.float32)).windowed(*dicom_windows.liver), cmap=plt.cm.bone);

# Plotting original data from NIfTI file
volume_nifti = nib.load('/kaggle/input/liver-tumor-segmentation/volume_pt1/volume-0.nii')
volume_data = volume_nifti.get_fdata()
slice_index = 0
plt.imshow(volume_data[..., slice_index], cmap='gray')
plt.title('Slice {} of Volume'.format(slice_index))
plt.axis('off')
plt.show()

# Plotting original mask from NIfTI file
volume_nifti = nib.load('/kaggle/input/liver-tumor-segmentation/segmentations/segmentation-10.nii')
volume_data = volume_nifti.get_fdata()
slice_index = 0
plt.imshow(volume_data[..., slice_index], cmap='gray')
plt.title('Slice {} of Volume'.format(slice_index))
plt.axis('off')
plt.show()

# Plotting volume and segmentation slices
volume_nifti = nib.load('/kaggle/input/liver-tumor-segmentation/volume_pt1/volume-0.nii')
segmentation_nifti = nib.load('/kaggle/input/liver-tumor-segmentation/segmentations/segmentation-0.nii')
volume_data = volume_nifti.get_fdata()
segmentation_data = segmentation_nifti.get_fdata()
slice_index = 0
plt.subplot(1, 2, 1)
plt.imshow(volume_data[..., slice_index], cmap='gray')
plt.title('Volume Slice {}'.format(slice_index))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation_data[..., slice_index], cmap='jet')
plt.title('Segmentation Slice {}'.format(slice_index))
plt.axis('off')
plt.show()

# Function to plot sample
def plot_sample(array_list, color_map='nipy_spectral'):
    fig = plt.figure(figsize=(20,16), dpi=100)

    plt.subplot(1, 4, 1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(tensor(array_list[0].astype(np.float32)).windowed(*dicom_windows.liver), cmap='bone');
    plt.title('Windowed Image')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Liver & Mask')
    plt.axis('off')

    plt.show()

# Plot sample
sample = 40
sample_slice = tensor(sample_ct[...,sample].astype(np.float32))
plot_sample([sample_ct[..., sample],
             sample_mask[..., sample]])

# Check mask values
mask = Image.fromarray(sample_mask[..., sample].astype('uint8'), mode="L")
unique, counts = np.unique(mask, return_counts=True)
print(np.array((unique, counts)).T)

# Preprocessing functions
class TensorCTScan(TensorImageBW): _show_args = {'cmap':'bone'}

@patch
def freqhist_bins(self:Tensor, n_bins=100):
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()

@patch
def freqhist_bin(self:Tensor, bins):
    x = (self[...,None]>=bins[...,:-1]).sum(0).float() / (bins[...,:-1].size(0))
    return x

@patch
def hist_scaled(self:Tensor, bins=None):
    if bins is None: bins = self.freqhist_bins()
    x = self.freqhist_bin(bins)
    return (x-x.min())/(x.max()-x.min())

# Datasets and DataLoaders
class LiverTumorSegmentationDataLoaders(DataLoaders):
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(cls, paths, get_items, label_func, valid_pct=0.2, seed=None, **kwargs):
        dblock = DataBlock(blocks=(ImageBlock(cls=TensorCTScan), MaskBlock(cls=TensorCTScan)),
                           get_items=get_items,
                           splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
                           get_y=label_func,
                           **kwargs)
        return cls.from_dblock(dblock, paths, **kwargs)

def get_items(noop):
    return df_files.index

def label_func(o):
    return df_files.loc[o, 'mask_dirname']+"/"+df_files.loc[o, 'mask_filename']

dls = LiverTumorSegmentationDataLoaders.from_label_func('/kaggle/input', get_items, label_func,
                                                        item_tfms=Resize(128),
                                                        batch_tfms=[*aug_transforms(flip_vert=True, size=128),
                                                                    Normalize.from_stats(*imagenet_stats)])

# Data augmentation
item_tfms = Resize(128)
batch_tfms = [*aug_transforms(flip_vert=True, size=128), Normalize.from_stats(*imagenet_stats)]

# Model training
learn = unet_learner(dls, resnet34, metrics=[Dice()], loss_func=CrossEntropyLossFlat(axis=1))

learn.fine_tune(10)

# Model interpretation
interp = SegmentationInterpretation.from_learner(learn)

# Plotting top losses
interp.plot_top_losses(k=3, figsize=(15,10))

# Plotting segmentation heatmap
interp.plot_top_losses(k=3, figsize=(15,10))

# Plotting confusion matrix
interp.plot_confusion_matrix(figsize=(7,7))
df.keras.models.save_model(model, 'unetmodel.h6')
