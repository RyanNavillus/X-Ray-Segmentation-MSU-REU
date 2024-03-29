#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append("../segmentation_models")
sys.path.append("../classification_models")

import collections
import cv2
import glob
import keras
import numpy as np
import os
import classification_models
import segmentation_models
import matplotlib.pyplot as plt
from scipy import io
from sklearn.preprocessing import normalize
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import Adam
from segmentation_models import Unet
from segmentation_models.segmentation_models.utils import set_trainable
from tqdm import tqdm

input_size = (512, 512)

def load_image(path):
    """Load grayscale image from path"""
    return cv2.resize(cv2.imread(path, 1), input_size)


def load_binary_image(path):
    """Load grayscale image from path"""
    return cv2.resize(cv2.imread(path, 0), input_size)


def load_mat(path):
    """Load grayscale image from path"""
    image = io.loadmat(path, appendmat=False)["dxImage"]["img"][0][0]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    colored_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
    return cv2.resize(colored_image, input_size)


def load_mask_mat(path):
    """Load grayscale image from path"""
    # import matlab files, extracting array of masks
    mask_list = np.stack(
        io.loadmat(path, appendmat=False)["maskImage"]["maskCrop"][0][0][0], axis=0
    )
    line_num = len(mask_list)

    # create placeholder arrays
    foreground = np.zeros((input_size[0], input_size[1], 1))
    background = np.ones((input_size[0], input_size[1], 1))

    # for each mask, scale it, reshape it, and add it to the foreground
    for i, mask in enumerate(mask_list):
        mask_array = cv2.resize(mask.astype(np.uint8), input_size)
        scaled_mask_array = np.reshape(mask_array, (input_size[0], input_size[1], 1))
        foreground = np.logical_or(foreground, scaled_mask_array)
    foreground = np.reshape(foreground, (1, input_size[0], input_size[1], 1))

    # create the background mask
    background = 1 - foreground

    # combine the background and foreground masks into a single array
    final_mask_list = np.array(np.append(background, foreground, axis=3))
    return final_mask_list


X = []
Y = []
data_dir = "../Masks/*.mat"
for filepath in tqdm(sorted(glob.glob(data_dir, recursive=True))):
    if filepath.endswith("image.mat"):
        X.append(load_mat(filepath))
    if filepath.endswith("mask.mat"):
        Y.append(load_mask_mat(filepath))
X = np.reshape(np.array(X), (-1, input_size[0], input_size[1], 3))
Y = np.reshape(np.array(Y), (len(X), input_size[0], input_size[1], 2))

line_ratio = 0
for image in Y[:, :, :, 0]:
    image = np.round(image, decimals=0)
    unique, counts = np.unique(image, return_counts=True)
    d = dict(zip(unique, counts))
    line_ratio += d[1.0] / (d[1.0] + d[0.0])
print(line_ratio / len(Y))


val_split = 0.15
test_split = 0.15
n = len(X)
sp1 = int(
    ((1 - val_split - test_split) * n) - 0.5
)  # Choose first index with rounding adjustment (0.5)
sp2 = int(
    ((1 - test_split) * n) - 0.5
)  # Choose second index with rounding adjustment (0.5)
X_train, Y_train = X[:sp1], Y[:sp1]
X_val, Y_val = X[sp1:sp2], Y[sp1:sp2]
X_test, Y_test = X[sp2:], Y[sp2:]
print(
    "{} Samples; {} train; {} val; {} test".format(
        n, len(X_train), len(X_val), len(X_test)
    )
)


smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true = y_true[:, :, :, 1]
    y_pred = y_pred[:, :, :, 1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def exact_dice_coef(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)

def cce(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
    loss_map = -K.sum(y_true * K.log(y_pred), axis=-1)
    return K.mean(K.flatten(loss_map))

def cce_plus_dice_coef_loss(y_true, y_pred):
    return 0.5*cce(y_true, y_pred) + 0.5*dice_coef_loss(y_true, y_pred)


# Create UNet model with custom resnet weights
model = Unet(
    "efficientnetb3",
    classes=2,
    input_shape=(512, 512, 3),
    encoder_weights="imagenet",
    activation="softmax",
)


model.summary()


model.compile(
    optimizer=Adam(lr=0.0001),
    loss=cce_plus_dice_coef_loss,
    metrics=["accuracy", dice_coef],
)


# Choose checkpoint path
checkpoint_path = "./UNET-with-EfficientNetB3_weights"
os.makedirs(checkpoint_path, exist_ok=True)

# Create early stopping callback
early_stopping = EarlyStopping(
    monitor="val_dice_coef", mode="max", patience=100, verbose=1
)

# Create checkpointer to save best model weights
checkpointer = ModelCheckpoint(
    filepath=checkpoint_path + "/efficientnetb3_cce_weights.h5",
    verbose=1,
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
)

# TEST: ReduceLROnPlateau
callback_list = [checkpointer, early_stopping]


res_history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=2000,
    verbose=1,
    callbacks=callback_list,
)
