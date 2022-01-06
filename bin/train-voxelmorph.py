#!/usr/bin/env python
# coding: utf-8

"""
Voxelmorph training script.
 
Largely inspired by `Voxelmorph's notebook tutorial <https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=Fw6dKBjBPXNp>`_,
and `Hypermorph training script <https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/train_hypermorph.py>`_.
"""

import os
import argparse

########################## line arguments ##########################

parser = argparse.ArgumentParser()

# general parameters
_models_path_default = os.path.abspath(str(__file__) + '/../../models')
parser.add_argument('--config', default='train-voxelmorph.json',
                    help='json file to use for extra configuration, like specifying which files to use for training (default: "train-voxelmorph.json")')
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
parser.add_argument('--batch-size', type=int, default=1,
                    help='training batch size, aka number of frames (default: 1)')
parser.add_argument('--gpu', default='',
                    help='visible GPU ID numbers. Goes into "CUDA_VISIBLE_DEVICES" env var (default: use all GPUs)')
parser.add_argument('--l2', type=float, default=0,
                    help='l2 regularization on the network weights (default: 0)')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--predict', action='store_true',
                    help='do not train, just load the model and predict on the dataset')
parser.add_argument('--ref', default='first',
                    help='reference frame to use when training. Either: first, last, mean or median (default: first)')
parser.add_argument('--validation-freq', type=int, default=10,
                    help='how often (in epochs) to calculate the validation loss (default: 10)')
parser.add_argument('--lr', type=float, nargs='+',
                    default=[1e-3, 1e-5],
                    help='learning rate. You can optionally provide this argument as'
                    ' `initial_lr final_lr` and exponential decay will be applied (default: 1e-3 1e-5)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 128 128)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 128 128 32 32 32 16 16)')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: ncc)')
parser.add_argument('--lambda-loss', type=float, default=0.5,
                    help='weighting for the losses. Image similarity loss will be weighted by `(1-lambda)` '
                    'while flow smoothness loss will be weighted by `lambda` (default: 0.5)')

args = parser.parse_args()

####################################################################

# validate arguments
if args.initial_epoch >= args.epochs:
    raise ValueError('--initial-epoch must be strictly lower than --epochs')

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ## Imports

import gc
import time
import json
import logging
from itertools import cycle
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

from stabilize2p.utils import make_video, get_strategy, vxm_data_generator

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

# read configuration file
config = defaultdict()
with open(args.config, 'r') as _config_f:
    config = json.load(_config_f)

np.random.seed(args.random_seed)

os.makedirs(args.out_dir, exist_ok=True)

strategy = get_strategy('GPU')

# retrieve dataset shape
in_shape = tiff.imread(config['training_pool'][0], key=0).shape

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 128, 128]
dec_nf = args.dec if args.dec else [128, 128, 32, 32, 32, 16, 16]

# weight save path
if args.model_dir.endswith('.h5'):
    save_filename = args.model_dir
else:
    save_filename = args.model_dir + '/vxm_drosophila_2d_{epoch:04d}.h5'


class L2Loss():
    def __init__(self, l2, kernel):
        self.l2 = l2
        self.kernel = kernel

    def __call__(self):
        return self.l2(self.kernel)


def train():
    # build model using VxmDense
    with strategy.scope():
        vxm_model = vxm.networks.VxmDense(in_shape, [enc_nf, dec_nf], int_steps=0)

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

    vxm_model.summary(line_length = 180)

    _LOGGER.info('input shape:  ' + ', '.join([str(t.shape) for t in vxm_model.inputs]))
    _LOGGER.info('output shape: ' + ', '.join([str(t.shape) for t in vxm_model.outputs]))

    # ## Loss

    # voxelmorph has a variety of custom loss classes
    losses = [None, vxm.losses.Grad('l2').loss]

    # prepare image similarity loss
    if args.image_loss == 'ncc':
        losses[0] = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        losses[0] = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


    # usually, we have to balance the two losses by a hyper-parameter
    loss_weights = [1-args.lambda_loss, args.lambda_loss]

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
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        metrics = get_metrics(optimizer)
        vxm_model.compile(optimizer=optimizer,
                          loss=losses,
                          loss_weights=loss_weights,
                          metrics=metrics)

    # data generators
    train_generator = vxm_data_generator(config['training_pool'], batch_size=args.batch_size, ref=args.ref)
    if config['validation_pool']:
        val_generator = vxm_data_generator(config['validation_pool'], batch_size=args.batch_size, ref=args.ref, training=False)
        # when training=False, the generator generates the frames once,
        # but tensorflow requires it to cycle in order to use it multiple
        # times
        val_generator = cycle(val_generator)

        nb_validation_frames = np.sum([len(tiff.TiffFile(file_path).pages) for file_path in config['validation_pool']])
        _LOGGER.info(f'{nb_validation_frames=}')
        validation_steps = (nb_validation_frames // args.batch_size)
    else:
        val_generator = None
        validation_steps = None

    # save model every 100 epochs
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=args.steps_per_epoch*100)

    _LOGGER.info(f'Training for {args.epochs} epochs')
    hist = vxm_model.fit(train_generator,
                         initial_epoch=args.initial_epoch,
                         epochs=args.epochs,
                         steps_per_epoch=args.steps_per_epoch,
                         validation_data=val_generator, validation_steps=validation_steps,
                         validation_freq=args.validation_freq,
                         callbacks=[save_callback],
                         verbose=1);

    # save last epoch
    vxm_model.save_weights(save_filename.format(epoch=args.epochs))


    # plot history
    import pickle
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
        # y-axis limits to see losses better
        losses = np.concatenate([l for l in hist.history.values()] + [[v] for v in config["reference_losses"].values()])
        low = np.min(losses)
        hig = np.quantile(losses, 0.90)
        ds = hig - low
        plt.ylim(low + 0.02*np.sign(low)*ds, hig + 0.01*np.sign(hig)*ds)
        # save plot
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
if not args.predict:
    train()


# ## Validation

# if there are validation tests
if config['validation_pool']:
    _LOGGER.info('starting validation..', flush=True)
    t1 = time.perf_counter()
    
    # build model using VxmDense
    with strategy.scope():
        vxm_model = vxm.networks.VxmDense(in_shape, [enc_nf, dec_nf], int_steps=0)
    
    # load weights
    path = save_filename.format(epoch=args.epochs)
    vxm_model.load_weights(path)
    _LOGGER.info(f'loaded model from: {path}', flush=True)
    
    # predict validation-set
    # TODO: make multiple validations possible!
    store_params = []
    val_generator = vxm_data_generator(config['validation_pool'][0],
                                       batch_size=args.batch_size,
                                       training=False,
                                       ref=args.ref,
                                       store_params=store_params)
    
    # TODO: concatenation is not a good idea for RAM usage!
    val_pred = []
    for (val_input, _) in val_generator:
        val_pred += [vxm_model.predict(val_input, verbose=2)]
    val_pred = [
        np.concatenate([a[0] for a in val_pred], axis=0),
        np.concatenate([a[1] for a in val_pred], axis=0)
    ]
    # can't do this or we run out of memory!
    # val_pred = vxm_model.predict(val_generator)

    # undo pre-processing
    params = store_params[0]
    h, l = params.pop('hig'), params.pop('low')
    val_pred[0] = val_pred[0] * (h - l) + l
    val_pred[0] = np.exp(val_pred[0]) - 1
    val_pred[0] = val_pred[0] + params['bg_thresh']

    t2 = time.perf_counter()
    _LOGGER.info(f'Predicted validation in {t2-t1:.2f}s | '
          f'{val_pred[0].shape[0]/(t2-t1):,.0f} frames/s | {(t2-t1)/val_pred[0].shape[0]:.4g} s/frame',
          flush=True)

    np.save(args.out_dir + '/validation-video', val_pred[0])
    np.save(args.out_dir + '/validation-flow', val_pred[1])
    
    # visualize flow
    i, j = val_pred[1].shape[1]//2, val_pred[1].shape[2]//2
    ne.plot.flow([val_pred[1][0, i:i+100, j:j+100, :].squeeze()], width=16, show=False);
    plt.savefig(args.out_dir + '/example-flow.svg')
    
    # generate validation video
    make_video(args.out_dir + '/validation-video', frame_gen(val_pred[0]))
    _LOGGER.info('generated validation video', flush=True)
