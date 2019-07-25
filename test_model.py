#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append("~/segmentation_models")
sys.path.append("~/classification_models")

import collections
import cv2
from datetime import datetime, time
import glob
import keras
import numpy as np
import pickle
import json
import os
import classification_models
import segmentation_models
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import io
from sklearn.preprocessing import normalize
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import Adam
from segmentation_models import Unet
from segmentation_models import FPN
from segmentation_models.utils import set_trainable
from segmentation_models.backbones import get_preprocessing
from segmentation_models.metrics import iou_score
from tqdm import tqdm


# Define Hyperparameters
learning_rate = 0.0005
decay = 0.0
optimizer = Adam(lr=learning_rate, decay=decay)
batch_size = 8
epochs = 1000
patience = 100
model_type = "unet"
model_weights = "imagenet"
model_backbone = "resnext101"
unet_filters = (256, 128, 64, 32, 16)
#unet_filters = (512, 256, 128, 64, 32)
fpn_dropout = 0.0
fpn_filters = 512
augmentation_list = None
val_split = 0.15
test_split = 0.15
loss = "unbalanced_weighted_bce_plus_dice_coef_loss"
early_stopping_metric = "val_loss"
early_stopping_mode = "min"
checkpoint_metric = "val_loss"
checkpoint_mode = "min"
input_size = (512, 512)
input_width = input_size[0]
input_height = input_size[1]
preprocess_input = get_preprocessing(model_backbone)

date_string = datetime.now().strftime("%m-%d-%y_%H:%M:%S")
epoch = datetime.utcfromtimestamp(0)
now = datetime.now()
uid = (now - epoch).total_seconds()
filename = str(uid)
print("Filename: {}".format(filename))

outfile_path = "/mnt/home/f0008576/Documents/Results/Training/" + filename + ".txt"
print("Writing to {}".format(outfile_path))
with open(outfile_path, "w") as outfile:
    outfile.write(
        "Date and Time: {}\n".format(datetime.now().strftime("%m/%d/%Y %H:%M %Ss"))
    )
    outfile.write("========== Model ==========\n")
    outfile.write("Model: {}\n".format(model_type))
    if model_type == "unet":
        outfile.write("Filter List: {}\n".format(unet_filters))
    elif model_type == "fpn":
        outfile.write("Filter count: {}\n".format(fpn_filters))
        outfile.write("Spatial Dropout Rate: {}\n".format(fpn_dropout))
    outfile.write("Backbone: {}\n".format(model_backbone))
    outfile.write("Weight Initialization: {}\n".format(model_weights))

    outfile.write("\n========== Data ==========\n")
    outfile.write(
        "Train/Validation/Test Split: {}/{}/{}\n".format(
            1-val_split-test_split, val_split, test_split
        )
    )
    outfile.write("Model: {}\n".format(model_type))

    outfile.write("\n========== Training ==========\n")
    outfile.write("Optimizer: {}\n".format(optimizer.__class__.__name__))
    outfile.write("Learning Rate: {}\n".format(learning_rate))
    outfile.write("Decay: {}\n".format(decay))
    outfile.write("Loss: {}\n".format(loss))
    outfile.write("Epochs: {}\n".format(epochs))
    outfile.write("Early Stopping Metric: {}\n".format(early_stopping_metric))
    outfile.write("Patience: {}\n".format(patience))
    outfile.write("Batch Size: {}\n\n".format(batch_size))


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
data_dir = "/mnt/home/f0008576/Masks/*.mat"
for filepath in tqdm(sorted(glob.glob(data_dir, recursive=True))):
    if filepath.endswith("image.mat"):
        X.append(load_mat(filepath))
    if filepath.endswith("mask.mat"):
        Y.append(load_mask_mat(filepath))
X = np.reshape(np.array(X), (-1, input_size[0], input_size[1], 3))
Y = np.reshape(np.array(Y), (len(X), input_size[0], input_size[1], 2))
print(len(X))
print(len(Y))

line_ratio = 0
for image in Y[:, :, :, 0]:
    image = np.round(image, decimals=0)
    unique, counts = np.unique(image, return_counts=True)
    d = dict(zip(unique, counts))
    line_ratio += d[1.0] / (d[1.0] + d[0.0])
print("Percentage of line pixels: {}%".format(100 * line_ratio / len(Y)))


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
    return (2.0 * intersection + epsilon) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + epsilon
    )


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weighted_bce(y_true, y_pred):
    output = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
    output = K.log(output / (1 - output))

    p = K.sum(K.flatten(y_true[..., :1])) / (224 ** 2)
    maximum_w = 1 / K.log(1.02 + 0)
    minimum_w = 1 / K.log(1.02 + 1)
    w = 1 / K.log(1.02 + p)
    scaled_w = (w - minimum_w) / (maximum_w - minimum_w)
    weighted_y_true = scaled_w * y_true

    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=weighted_y_true, logits=output
    )


def cce(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
    loss_map = -K.sum(y_true * K.log(y_pred), axis=-1)
    return K.mean(K.flatten(loss_map))


def cce_plus_dice_coef_loss(y_true, y_pred):
    return 0.5 * cce(y_true, y_pred) + 0.5 * dice_coef_loss(y_true, y_pred)


def weighted_bce_plus_dice_coef_loss(y_true, y_pred):
    return 0.5 * weighted_bce(y_true, y_pred) + 0.5 * dice_coef_loss(y_true, y_pred)


def unbalanced_weighted_bce_plus_dice_coef_loss(y_true, y_pred):
    return 0.1 * weighted_bce(y_true, y_pred) + 0.9 * dice_coef_loss(y_true, y_pred)


# Create UNet model with custom resnet weights
model = None
if model_type == "unet":
    model = Unet(
        model_backbone,
        classes=2,
        input_shape=(input_size[0], input_size[1], 3),
        encoder_weights=model_weights,
        decoder_filters=unet_filters,
        activation="softmax",
    )
elif model_type == "fpn":
    model = FPN(
        model_backbone,
        classes=2,
        input_shape=(input_size[0], input_size[1], 3),
        encoder_weights=model_weights,
        pyramid_block_filters=fpn_filters,
        pyramid_dropout=fpn_dropout,
        activation="softmax",
    )


model.summary()

model.compile(
    optimizer=optimizer,
    loss=globals()[loss],
    metrics=["accuracy", dice_coef, segmentation_models.metrics.iou_score],
)


# Choose checkpoint path
checkpoint_path = "./Weights/"
os.makedirs(checkpoint_path, exist_ok=True)

# Create early stopping callback
early_stopping = EarlyStopping(
    monitor=early_stopping_metric, mode=early_stopping_mode, patience=patience, verbose=1
)

weights_filepath = checkpoint_path + filename + ".h5"
print("Saving weights to {}".format(weights_filepath))

# Create checkpointer to save best model weights
checkpointer = ModelCheckpoint(
    filepath=weights_filepath,
    verbose=1,
    monitor=checkpoint_metric,
    mode=checkpoint_mode,
    save_best_only=True,
)

callback_list = [checkpointer, early_stopping]

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callback_list,
)

# Save history
history_path = "/mnt/home/f0008576/Documents/Results/History/" + filename
print("Writing history dictionary to {}".format(history_path))
with open(history_path, 'w') as history_file:
    json.dump(history.history, history_file)

# Evaluate model
model.load_weights(weights_filepath)
print("X test n = {}".format(len(X_test)))
print("Y test n = {}".format(len(Y_test)))
loss, accuracy, dice, iou = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', loss)
print('Test accuracy: ', accuracy)
print('Dice coef: ', dice)
print('IoU: ', iou)

with open(outfile_path, "a") as outfile:
    outfile.write('Test loss: {}\n'.format(loss))
    outfile.write('Test accuracy: {}\n'.format(accuracy))
    outfile.write('Dice coef: {}\n'.format(dice))
    outfile.write('IoU: {}\n'.format(iou))


def print_sample(sample, uid, test_idx, title=None, figsize=(10,10)):
    plt.ioff()
    sample = np.squeeze(sample)
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.imshow(sample, cmap='gray')
    plt.savefig("/mnt/home/f0008576/Documents/Results/Sample_Predictions/" + filename + "_sample" + str(test_idx) + ".png")
    plt.close()


def print_mask(sample, mask, uid, test_idx, title=None, prediction=True, figsize=(10,10)):
    plt.ioff()
    mask = np.squeeze(mask)
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.imshow(sample, cmap='gray')
    plt.imshow(np.ma.masked_where(mask[:, :, 1] <= 0.5, mask[:, :, 1]), cmap='jet', alpha=0.5, vmin=0, vmax=1)
    file_descriptor = ""
    if prediction:
        file_descriptor = "_prediction-"
    else:
        file_descriptor = "_truth-"
    plt.savefig("/mnt/home/f0008576/Documents/Results/Sample_Predictions/" + filename + file_descriptor + str(test_idx) + ".png")
    plt.close()


for test_id in range(10):
    predicted_mask = model.predict(np.expand_dims(X_test[test_id], axis=0))
    sample_iou = iou_score(Y_test[test_id], predicted_mask)
    print_mask(X_test[test_id], predicted_mask, uid, test_id, title="Prediction for image {}\nIoU = {}".format(str(test_id), sample_iou), prediction=True, figsize=(10,10))
    #print_mask(X_test[test_id], Y_test[test_id], uid, test_id, title="True mask for image {}".format(str(test_id), prediction=False, figsize=(10,10)))


print(filename)
