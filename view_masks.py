#!/usr/bin/env python
import cv2
import glob
import matplotlib
import numpy as np
import sys
from scipy import io
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

input_size = (512, 512)

def load_mat(path):
    """Load grayscale image from path"""
    image = io.loadmat(path, appendmat=False)['dxImage']['img'][0][0]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    colored_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
    return cv2.resize(colored_image, input_size)


def load_mask_mat(path):
    """Load grayscale image from path"""
    # import matlab files, extracting array of masks
    mask_list = np.stack(io.loadmat(path, appendmat=False)['maskImage']['maskCrop'][0][0][0], axis=0)
    line_num = len(mask_list)

    # create placeholder arrays
    foreground = np.zeros((input_size[0], input_size[1], 1))
    background = np.ones((1, input_size[0], input_size[1], 1))

    # for each mask, scale it, reshape it, and add it to the foreground
    for i, mask in enumerate(mask_list):
        mask_array = cv2.resize(mask.astype(np.uint8), input_size)
        scaled_mask_array = np.reshape(mask_array, (input_size[0], input_size[1], 1))
        foreground[np.where(scaled_mask_array!=0)] = i+1
    foreground = np.reshape(foreground, (1, input_size[0], input_size[1], 1))

    # create the background mask
    background[np.where(foreground!=0)] = 0

    # combine the background and foreground masks into a single array
    final_mask_list = np.array(np.append(background, foreground, axis=3))
    return final_mask_list

def plot_mask(sample, mask, figsize=(10,10), title=""):
    mask = np.squeeze(mask)
    plt.figure(figsize=figsize)
    plt.imshow(sample, cmap='gray')
    plt.imshow(np.ma.masked_where(mask[:, :, 1] <= 0, mask[:, :, 1]), cmap='prism', alpha=0.1)
    plt.tight_layout()
    plt.title(title)
    plt.show()

# Display all images matching the path below
for filename in sorted(glob.glob("/data/midi-lab-general/osemis_annotations/osemis_annotation_file_to_masks/Masks/*_crop_image.mat")):
    mask_path = filename[:-15] + "_line_mask.mat"
    print(filename)
    print(mask_path)
    sample = load_mat(filename)
    mask = load_mask_mat(mask_path)

    plot_mask(sample, mask, title=filename[80:-15])
