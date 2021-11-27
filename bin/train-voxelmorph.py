#!/usr/bin/env python
# coding: utf-8

# # Voxelmorph training Notebook
# 
# Largely inspired by [Voxelmorph's notebook tutorial](https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=Fw6dKBjBPXNp).


import os

# so we can use the modules/ folder
import sys
sys.path.insert(0, '../')

import argparse

########################## line arguments ##########################

parser = argparse.ArgumentParser()

# general parameters
_models_path_default = os.path.abspath(str(__file__) + '/../../models')
parser.add_argument('--model-dir', default=_models_path_default,
                    help=f'model output directory (default: {_models_path_default})')
parser.add_argument('--out-dir', default='train-voxelmorph.out',
                    help='output directory for plots and predictions (default: "train-voxelmorph.out/")')
parser.add_argument('--random-seed', type=int, default=1,
                    help='numpy\'s random seed (default: 1)')

# training parameters
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs (default: 200)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='steps per epoch (default: 100)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='training batch size, aka number of frames (default: 8)')
parser.add_argument('--l2', type=float, default=1e-4,
                    help='l2 regularization on the network weights (default: 1e-4)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--gpu', default='', help='visible GPU ID numbers. Goes into "CUDA_VISIBLE_DEVICES" env var (default: use all GPUs)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')

args = parser.parse_args()

####################################################################


if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from collections import defaultdict

import cv2
import tifffile as tiff
import numpy as np
import matplotlib
import voxelmorph as vxm
import tensorflow as tf
import neurite as ne
from matplotlib import pyplot as plt
from sklearn.utils import gen_batches

from modules.utils import make_video, get_strategy, vxm_data_generator

# matplotlib Font size
plt.style.use('default')

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def frame_gen(video, scores=None, lt=0.9):
    std = video[0].std()
    print('calculated std')
    low = np.quantile(video[0], q=0.01)
    if scores is not None:
        for img, score in zip(video, scores):
            img = (img - low) / std * 255 / 3
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype(np.uint8)
            if score < lt:
                img[:50, :50] = 255
            else:
                img[:50, :50] = 0
            yield img
    else:
        for img in video:
            img = (img - low) / std * 255 / 3
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype(np.uint8)
            yield img


# read configuration file
config = defaultdict()
with open('train-voxelmorph.json') as _config_f:
    config = json.load(_config_f)


# ## Fully-Conv NN

np.random.seed(args.random_seed)

os.makedirs(args.out_dir, exist_ok=True)

strategy = get_strategy('GPU')

# retrieve dataset shape
in_shape = tiff.imread(config['training_pool'][0], key=0).shape

# configure unet features 
nb_features = [
    [32, 32, 32, 64],         # encoder features
    [64, 32, 32, 16, 16, 16]  # decoder features
]

# build model using VxmDense
with strategy.scope():
    vxm_model = vxm.networks.VxmDense(in_shape, nb_features, int_steps=0)


# After loading our pre-trained model, we are going loop over all of its layers.
# For each layer, we check if it supports regularization, and if it does, we add it
if args.l2:
    regularizer = tf.keras.regularizers.l2(args.l2)
    for layer in vxm_model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

# load initial weights (if provided)
if args.load_weights:
    vxm_model.load_weights(args.load_weights)

vxm_model.summary(line_length = 180)

print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))


# ## Loss




# voxelmorph has a variety of custom loss classes
losses = [None, vxm.losses.Grad('l2').loss]

# prepare image loss
if args.image_loss == 'ncc':
    losses[0] = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    losses[0] = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]


# ## Compile model

with strategy.scope():
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)


# ## Train


# data generators
train_generator = vxm_data_generator(config['training_pool'], batch_size=args.batch_size)
val_generator = vxm_data_generator(config['validation_pool'], batch_size=args.batch_size)

nb_validation_frames = np.sum([len(tiff.TiffFile(file_path).pages) for file_path in config['validation_pool']])
print(f'{nb_validation_frames=}')

# training
save_filename = args.model_dir + '/vxm_drosophila_2d_{epoch:04d}.h5'

save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=args.steps_per_epoch*100)

print(f'Training for {args.epochs} epochs')
hist = vxm_model.fit(train_generator,
                     epochs=args.epochs,
                     steps_per_epoch=args.steps_per_epoch,
                     validation_data=val_generator, validation_steps=(nb_validation_frames // args.batch_size),
                     callbacks=[save_callback],
                     verbose=1);


vxm_model.save_weights(save_filename.format(epoch=args.epochs))


# plot history

import matplotlib.pyplot as plt

def plot_history(hist, loss_names=['loss', 'val_loss']):
    # Simple function to plot training history.
    plt.figure(figsize=(10, 6))
    for loss_name in loss_names:
        plt.plot(hist.epoch, hist.history[loss_name], '.-', label=loss_name)
    plt.ylabel(args.image_loss + ' loss')
    plt.xlabel('epoch')
    plt.title('Training loss')
    plt.legend()
    plt.savefig(args.out_dir + '/history.svg')

plot_history(hist)

# no more need for training set
del train_generator


# ## Validation


# configure unet features 
nb_features = [
    [32, 32, 32, 64],         # encoder features
    [64, 32, 32, 16, 16, 16]  # decoder features
]

# build model using VxmDense
with strategy.scope():
    vxm_model = vxm.networks.VxmDense(in_shape, nb_features, int_steps=0)


path = save_filename.format(epoch=args.epochs)
vxm_model.load_weights(path)
print(f'loaded model from: {path}')


# validation

val_generator = vxm_data_generator(config['validation_pool'][0], batch_size=args.batch_size, training=False)

val_pred = []
for (val_input, _) in val_generator:
    val_pred += [vxm_model.predict(val_input, verbose=2)]
val_pred = [
    np.concatenate([a[0] for a in val_pred], axis=0),
    np.concatenate([a[1] for a in val_pred], axis=0)
]

np.save(args.out_dir + '/validation-video', val_pred[0])
np.save(args.out_dir + '/validation-flow', val_pred[1])


# visualize flow

i, j = val_pred[1].shape[1]//2, val_pred[1].shape[2]//2
ne.plot.flow([val_pred[1][0, i:i+100, j:j+100, :].squeeze()], width=16, show=False);
plt.savefig(args.out_dir + '/example-flow.svg')

# generate validation video

make_video(args.out_dir + '/validation-video', frame_gen(val_pred[0]))
print('generated validation video')
