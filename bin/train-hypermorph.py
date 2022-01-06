#!/usr/bin/env python
# coding: utf-8
"""Voxelmorph training script.
 
Largely inspired by `Voxelmorph's notebook tutorial <https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=Fw6dKBjBPXNp>`_,
and `Hypermorph training script <https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/train_hypermorph.py>`_.

Unfortunately, GPUs are NOT supported!
"""

import os
import argparse

########################### line arguments ###########################

parser = argparse.ArgumentParser()

# general parameters
_models_path_default = os.path.abspath(str(__file__) + '/../../models')
parser.add_argument('--config', default='train-hypermorph.json',
                    help='json file to use for extra configuration, like specifying which files to use for training (default: "train-hypermorph.json")')
parser.add_argument('--model-dir', default=_models_path_default,
                    help='model output directory. Can be the full .h5 path of the model that will be saved, '
                         'using Keras formats like \'{epoch:04d}\' 'f' (default: {_models_path_default})')
parser.add_argument('--out-dir', default='train-hypermorph.out',
                    help='output directory for plots and predictions (default: "train-hypermorph.out/")')
parser.add_argument('--random-seed', type=int, default=1,
                    help='numpy\'s random seed (default: 1)')

# training parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs (default: 1000)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='steps per epoch (default: 100)')
parser.add_argument('--batch-size', type=int, default=1,
                    help='training batch size, aka number of frames (default: 1)')
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--l2', type=float, default=0,
                    help='l2 regularization on the network weights (default: 0)')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--ref', default='first',
                    help='reference frame to use when training. Either: first, last, mean or median (default: first)')
parser.add_argument('--validation-freq', type=int, default=50,
                    help='how often (in epochs) to calculate the validation loss. Set to -1 to disable (default: 50)')
parser.add_argument('--validation-hyp', type=float, default=0.5,
                    help='what hyper-parameter value to use in validation. Should be in the interval [0, 1] (default: 0.5)')
parser.add_argument('--lr', type=float, nargs='+',
                    default=[1e-3, 1e-5],
                    help='learning rate. You can optionally provide this argument as'
                    ' `initial_lr final_lr` and exponential decay will be applied (default: 1e-3 1e-5)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--oversample-rate', type=float, default=0.2,
                    help='hyperparameter end-point over-sample rate (default 0.2)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: ncc)')
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')

args = parser.parse_args()

assert len(args.lr) <= 2, '--lr should be 1 or 2 arguments'

######################################################################

# validate arguments
if args.initial_epoch >= args.epochs:
    raise ValueError('--initial-epoch must be strictly lower than --epochs')

# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ## Imports

import gc
import time
import json
import pickle
import logging
from itertools import cycle
from collections import defaultdict

import cv2
import tifffile as tiff
import numpy as np
import matplotlib
import voxelmorph as vxm
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import neurite as ne
from matplotlib import pyplot as plt
from sklearn.utils import gen_batches
from tensorflow.keras import backend as K

from stabilize2p.utils import make_video, get_strategy, \
                              vxm_data_generator, hypermorph_dataset, \
                              hypermorph_optimal_register

# logger
_LOGGER = logging.getLogger('stabilize2p')

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

# reduce logging in the package to only show errors
# logging.getLogger('stabilize2p').setLevel(logging.ERROR)


def frame_gen(video):
    low, hig = video[0].min(), video[1].max()
    for img in video:
        img = (img - low) / (hig - low) * 255
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        yield img

# ## Setup

_LOGGER.info('script args: ' + str(vars(args)))

# solves Hypermorph's compatibility issues
tf1.disable_eager_execution()
tf1.experimental.output_all_intermediates(True)

# read configuration file
config = defaultdict()
with open(args.config, 'r') as _config_f:
    config = json.load(_config_f)

np.random.seed(args.random_seed)

os.makedirs(args.out_dir, exist_ok=True)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
assert np.mod(args.batch_size, nb_devices) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# weight save path
if args.model_dir.endswith('.h5'):
    save_filename = args.model_dir
else:
    save_filename = args.model_dir + '/hypm_drosophila_2d_{epoch:04d}.h5'

# data generators
_LOGGER.info('Initializing generators..')
t1 = time.perf_counter()

inshape = tiff.imread(config['training_pool'][0], key=0).shape
nfeats = 1

base_train_generator = vxm_data_generator(
    config['training_pool'],
    batch_size=args.batch_size,
    ref=args.ref)
train_dataset = hypermorph_dataset(base_train_generator,
                                   train=True,
                                   inshape=inshape,
                                   nfeats=nfeats,
                                   oversample_rate=args.oversample_rate)

if config['validation_pool'] and args.validation_freq > 0:
    base_val_generator = vxm_data_generator(
        config['validation_pool'],
        # don't make it _too_ slow
        batch_size=max(8, args.batch_size),
        ref=args.ref, training=False)
    # when training=False, the generator generates the frames once,
    # but tensorflow requires it to cycle in order to use it multiple times
    base_val_generator = cycle(base_val_generator)
    val_dataset = hypermorph_dataset(base_val_generator,
                                     train=False,
                                     inshape=inshape,
                                     nfeats=nfeats,
                                     hyp_value=args.validation_hyp)

    nb_validation_frames = np.sum([len(tiff.TiffFile(file_path).pages) for file_path in config['validation_pool']])
    validation_steps = (nb_validation_frames // args.batch_size)
    # force garbage collector
    gc.collect()
    _LOGGER.info(f'{nb_validation_frames=}')
else:
    val_dataset = None
    validation_steps = None
t2 = time.perf_counter()
_LOGGER.info(f'Initialized generators in {t2-t1:.2f}s')

# # (SLOW) extract shape and number of features from sampled input
# sample_shape = next(train_dataset)[0][0].shape
# inshape = sample_shape[1:-1]
# nfeats = sample_shape[-1]


class L2Loss():
    def __init__(self, l2, kernel):
        self.l2 = l2
        self.kernel = kernel

    def __call__(self):
        return self.l2(self.kernel)


def train():
    # build model using VxmDense
    vxm_model = vxm.networks.HyperVxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
        svf_resolution=2,
    )

    # load initial weights (if provided)
    if args.load_weights:
        vxm_model.load_weights(args.load_weights)

    # After loading our pre-trained model, we are going loop over all of its layers.
    # For each layer, we check if it supports regularization, and if it does, we add it
    if args.l2:
        for li, layer in enumerate(vxm_model.layers):
            if hasattr(layer, 'kernel_regularizer') and hasattr(layer, 'kernel'):
                l2 = tf.keras.regularizers.l2(args.l2)
                layer.kernel_regularizer = l2
                layer.add_loss(L2Loss(l2, layer.kernel))
    
        _LOGGER.info('losses:' + str(vxm_model.losses))
        assert len(vxm_model.losses) > 0, 'Could not apply l2 regularization'

    vxm_model.summary(line_length=180)

    _LOGGER.info('input shape:  ' + ', '.join([str(t.shape) for t in vxm_model.inputs]))
    _LOGGER.info('output shape: ' + ', '.join([str(t.shape) for t in vxm_model.outputs]))
    
    # save script's arguments in the model
    # vxm_model.references.script_args = vars(args)
    # _LOGGER.info('storing args in the model:' + str(vars(args)))

    # ## Loss

    # prepare image similarity loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        scaling = 1.0 / (args.image_sigma ** 2)
        image_loss_func = lambda x1, x2: scaling * K.mean(K.batch_flatten(K.square(x1 - x2)), -1)
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # prepare loss functions and compile model
    def image_loss(y_true, y_pred):
        hyp = (1 - tf.squeeze(vxm_model.references.hyper_val))
        return hyp * image_loss_func(y_true, y_pred)

    def grad_loss(y_true, y_pred):
        hyp = tf.squeeze(vxm_model.references.hyper_val)
        return hyp * vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss(y_true, y_pred)

    # multi-gpu support
    if nb_devices > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
        vxm_model = tf.keras.utils.multi_gpu_model(vxm_model, gpus=nb_devices)
    else:
        # save model every 100 epochs
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=args.steps_per_epoch*100)

    # ## Compile model
    if len(args.lr) == 1:
        learning_rate = args.lr[0]
        get_metrics = lambda _: None
    else:
        def get_metrics(optimizer):
            def lr(y_true, y_pred):
                return optimizer._decayed_lr(tf.float32)
            return [lr]

        # lr scheduler
        lr_decay_factor = (args.lr[1] / args.lr[0])**(1/args.epochs)
        nb_train_frames = np.sum([len(tiff.TiffFile(path).pages) for path in config['training_pool']])
        gc.collect()
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.lr[0],
                decay_steps=int(nb_train_frames/args.batch_size),
                decay_rate=lr_decay_factor,
                staircase=True)

    # ## Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = get_metrics(optimizer)
    vxm_model.compile(optimizer=optimizer, loss=[image_loss, grad_loss], metrics=metrics)

    _LOGGER.info(f'Training for {args.epochs} epochs')
    hist = vxm_model.fit(train_dataset,
                         initial_epoch=args.initial_epoch,
                         epochs=args.epochs,
                         steps_per_epoch=args.steps_per_epoch,
                         validation_data=val_dataset, validation_steps=validation_steps,
                         validation_freq=args.validation_freq,
                         callbacks=[save_callback],
                         verbose=1);

    # save last epoch
    vxm_model.save_weights(save_filename.format(epoch=args.epochs))


    # plot history
    import matplotlib.pyplot as plt

    def plot_history(hist, loss_names=['loss', 'val_loss']):
        # Simple function to plot training history.
        plt.figure(figsize=(10, 6))

        color = plt.cm.tab10(np.linspace(0, 1, 10))
        c = cycle(color)

        # training losses
        for loss_name in loss_names:
            if loss_name in hist.history:
                # crude way to retrieve loss calculation frequency
                N = len(hist.epoch)
                M = len(hist.history[loss_name])
                freq = N // M

                plt.plot(np.arange(freq, N+1, freq)[:M],
                         hist.history[loss_name], '.-', c=next(c), label=loss_name)

        # user-provided reference losses
        for label, value in config["reference_losses"].items():
            plt.axhline(value, label=label, ls='--', c=next(c))

        plt.ylabel(args.image_loss + '+flow loss')
        plt.xlabel('epoch')
        plt.title('Training loss')
        plt.legend()
        plt.savefig(args.out_dir + '/history.svg')


    # save to file
    with open(args.out_dir + '/history.pkl', 'wb') as file:
        pickle.dump({'epoch': hist.epoch, 'history': hist.history}, file)
        _LOGGER.info('saved history')

    plot_history(hist)
    _LOGGER.info('plotted history')

    # no more need for the model
    del vxm_model
    # clear Keras session
    tf.keras.backend.clear_session()


# ## Training
train()