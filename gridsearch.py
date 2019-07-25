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
import tensorflow as tf
from scipy import io
from sklearn.preprocessing import normalize
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
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


epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)

def dice_coef(y_true, y_pred):
    y_true = y_true[:, :, :, 1]
    y_pred = y_pred[:, :, :, 1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def weighted_bce(y_true, y_pred):
    output = K.clip(y_pred, 1e-7, 1. - 1e-7)
    output = K.log(output / (1 - output))

    p = K.sum(K.flatten(y_true[...,:1])) / (224**2)
    maximum_w = 1 / K.log(1.02 + 0)
    minimum_w = 1 / K.log(1.02 + 1)
    w = 1 / K.log(1.02 + p)
    scaled_w = (w - minimum_w) / (maximum_w - minimum_w)
    weighted_y_true = scaled_w * y_true

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=weighted_y_true, logits=output)

def cce(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
    loss_map = -K.sum(y_true * K.log(y_pred), axis=-1)
    return K.mean(K.flatten(loss_map))

def cce_plus_dice_coef_loss(y_true, y_pred):
    return 0.5*cce(y_true, y_pred) + 0.5*dice_coef_loss(y_true, y_pred)

def weighted_bce_plus_dice_coef_loss(y_true, y_pred):
    return 0.5*weighted_bce(y_true, y_pred) + 0.5*dice_coef_loss(y_true, y_pred)

def unbalanced_weighted_bce_plus_dice_coef_loss(y_true, y_pred):
    return 0.1*weighted_bce(y_true, y_pred) + 0.9*dice_coef_loss(y_true, y_pred)


# Hyper parameters

learning_rate = np.random.uniform(0.00001, 0.001)
batch_size = np.random.randint(1,5) * 2
optimizer_choice = np.random.randint(0,3)
loss_choice = np.random.randint(0,3)
optimizers = {
    "Adam": Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    "SGD": SGD(lr=learning_rate, momentum=0.9),
    "RMSprop": RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
}
loss_fns = {
    "cce": cce,
    "weighted_bce": weighted_bce,
    "dice_loss": dice_coef_loss,
    "cce+dice_loss": cce_plus_dice_coef_loss,
    "weighted_bce+dice_loss": weighted_bce_plus_dice_coef_loss,
    "unbalanced_weighted_bce+dice_loss": unbalanced_weighted_bce_plus_dice_coef_loss,
}
optimizer = optimizers[list(optimizers.keys())[optimizer_choice]]
loss = loss_fns[list(loss_fns.keys())[loss_choice]]
print("Learning Rate: {}".format(learning_rate))
print("Batch Size: {}".format(batch_size))
print("Optimizer: {}".format(list(optimizers.keys())[optimizer_choice]))
print("Loss Function: {}".format(list(loss_fns.keys())[loss_choice]))


# Create UNet model with custom resnet weights
model = Unet(
    "efficientnetb3",
    classes=2,
    input_shape=(512, 512, 3),
    encoder_weights=None,
    activation="softmax",
)


model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy", dice_coef],
)


# Create early stopping callback
early_stopping = EarlyStopping(
    monitor="val_dice_coef", mode="max", patience=50, verbose=1
)

callback_list = [early_stopping]

res_history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    batch_size=batch_size,
    epochs=500,
    verbose=1,
    callbacks=callback_list,
)

# Evaluate Model
loss, accuracy, dice = model.evaluate(X_test, Y_test, verbose=0)
n_epochs = len(res_history.history['loss'])
print('Number of Epochs: {}'.format(n_epochs))
print('Test loss: {:.5f}'.format(loss))
print('Test accuracy: {:.5f}'.format(accuracy))
print('Dice coef: {:.5f}'.format(dice))

# TODO: Create file and save hyperparameters and test results
hp_path = "./Hyperparameter-Search-Fixed/"
os.makedirs(hp_path, exist_ok=True)

hp_file = open(os.path.join(hp_path, "{:.10f}.txt".format(learning_rate)), "w")
hp_file.write("Learning Rate: {:7f}\n".format(learning_rate))
hp_file.write("Batch Size: {}\n".format(batch_size))
hp_file.write("Optimizer: {}\n".format(list(optimizers.keys())[optimizer_choice]))
hp_file.write("Loss Function: {}\n".format(list(loss_fns.keys())[loss_choice]))
hp_file.write('\n--- Results ---\n')
hp_file.write('Number of Epochs: {}'.format(n_epochs))
hp_file.write('Test loss: {:.5f}\n'.format(loss))
hp_file.write('Test accuracy: {:.5f}\n'.format(accuracy))
hp_file.write('Dice coef: {:.5f}\n'.format(dice))
hp_file.close()
