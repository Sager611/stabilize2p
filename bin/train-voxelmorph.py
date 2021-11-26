#!/usr/bin/env python
# coding: utf-8

# # Voxelmorph training Notebook
# 
# Largely inspired by [Voxelmorph's notebook tutorial](https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=Fw6dKBjBPXNp).

# In[1]:


import os

# so we can use the modules/ folder
import sys
sys.path.insert(0, '../')

import argparse

########################## line arguments ##########################

parser = argparse.ArgumentParser()

# general parameters
_models_path_default = os.path.abspath('../models')
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


import cv2
import tifffile as tiff
import numpy as np
import matplotlib
import voxelmorph as vxm
import tensorflow as tf
import neurite as ne
from matplotlib import pyplot as plt

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


# In[2]:


from modules.utils import make_video

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


# In[3]:


orig_examples = [
    '../data/200901_G23xU1/Fly1/Fly1/001_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/002_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/003_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/004_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/005_coronal/2p/denoised_red.tif'
]

warped_examples = [
    '../data/200901_G23xU1/Fly1/Fly1/001_coronal/2p/warped_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/002_coronal/2p/warped_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/003_coronal/2p/warped_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/004_coronal/2p/warped_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/005_coronal/2p/warped_red.tif'
]

pystackreg_examples = [
    '../data/200901_G23xU1/Fly1/Fly1/001_coronal/2p/denoised_red.pystackreg-affine.tif',
    '../data/200901_G23xU1/Fly1/Fly1/002_coronal/2p/denoised_red.pystackreg-affine.tif',
    '../data/200901_G23xU1/Fly1/Fly1/003_coronal/2p/denoised_red.pystackreg-affine.tif',
    '../data/200901_G23xU1/Fly1/Fly1/004_coronal/2p/denoised_red.pystackreg-affine.tif',
    '../data/200901_G23xU1/Fly1/Fly1/005_coronal/2p/denoised_red.pystackreg-affine.tif'
]


# In[4]:


from sklearn.utils import gen_batches


def vxm_data_generator(x_data, batch_size=32, training=True):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    # fixed image reference frame is the initial frame
    idx2 = np.zeros(batch_size, dtype=int)

    if training:
        while True:
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, 1]
            idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
            moving_images = x_data[idx1, ..., np.newaxis]
            # idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
            fixed_images = x_data[idx2, ..., np.newaxis]
            inputs = [moving_images, fixed_images]

            # prepare outputs (the 'true' moved image):
            # of course, we don't have this, but we know we want to compare 
            # the resulting moved image with the fixed image. 
            # we also wish to penalize the deformation field. 
            outputs = [fixed_images, zero_phi]

            yield (inputs, outputs)
    else:
        for idx in gen_batches(x_data.shape[0], batch_size):
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, 1]
            moving_images = x_data[idx, ..., np.newaxis]
            fixed_images = x_data[idx2[:moving_images.shape[0]], ..., np.newaxis]
            inputs = [moving_images, fixed_images]

            # prepare outputs (the 'true' moved image):
            # of course, we don't have this, but we know we want to compare 
            # the resulting moved image with the fixed image. 
            # we also wish to penalize the deformation field. 
            outputs = [fixed_images, zero_phi]

            yield (inputs, outputs)


# In[5]:


def preprocessing(x):
    # normalize
    low, hig = x.min(), x.max()
    x = (x - low) / (hig - low)
    return x, (low, hig)


# ## Fully-Conv NN

np.random.seed(args.random_seed)

os.makedirs(args.out_dir, exist_ok=True)

# In[6]:


from modules.utils import get_strategy
strategy = get_strategy('GPU')


# In[7]:


# retrieve dataset shape
PATH = orig_examples[0]
in_shape = tiff.imread(PATH, key=0).shape


# In[8]:


# configure unet features 
nb_features = [
    [32, 32, 32, 64],         # encoder features
    [64, 32, 32, 16, 16, 16]  # decoder features
]

# build model using VxmDense
with strategy.scope():
    vxm_model = vxm.networks.VxmDense(in_shape, nb_features, int_steps=0)


# In[9]:


vxm_model.summary(line_length = 180)


# In[10]:


print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))


# ## Loss

# In[11]:


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

# In[12]:


with strategy.scope():
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)


# ## Train

# In[13]:


x_train = tiff.imread(orig_examples[0], key=range(200))
x_train, (x_train_low, x_train_hig) = preprocessing(x_train)

# let's test it
train_generator = vxm_data_generator(x_train, batch_size=args.batch_size)
in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample] 
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['viridis', 'viridis', 'viridis', 'gray'], do_colorbars=True);

# validation
x_val = tiff.imread(orig_examples[1], key=range(200))
x_val = (x_val - x_train_low) / (x_train_hig - x_train_low)
print(f'{x_val.min()=} | {x_val.max()=}')

# let's get some data
val_generator = vxm_data_generator(x_val, batch_size=args.batch_size)


# training,

# In[ ]:

save_filename = args.model_dir + '/vxm_drosophila_2d_{epoch:04d}.h5'

save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=100)

print(f'Training for {args.epochs} epochs')
hist = vxm_model.fit(train_generator,
                     epochs=args.epochs,
                     steps_per_epoch=args.steps_per_epoch,
                     validation_data=val_generator, validation_steps=(x_val.shape[0] // args.batch_size),
                     callbacks=[save_callback],
                     verbose=1);


vxm_model.save_weights(save_filename.format(epoch=args.epochs))


# In[ ]:


import matplotlib.pyplot as plt

def plot_history(hist, loss_names=['loss', 'val_loss']):
    # Simple function to plot training history.
    plt.figure(figsize=(8, 6))
    for loss_name in loss_names:
        plt.plot(hist.epoch, hist.history[loss_name], '.-', label=loss_name)
    plt.ylabel(args.image_loss + ' loss')
    plt.xlabel('epoch')
    plt.title('Training loss')
    plt.legend()
    plt.savefig(args.out_dir + '/history.svg')

plot_history(hist)


# In[ ]:


# no more need for training set
del x_train, x_train_low, x_train_hig, train_generator


# ## Validation


# In[ ]:


# configure unet features 
nb_features = [
    [32, 32, 32, 64],         # encoder features
    [64, 32, 32, 16, 16, 16]  # decoder features
]

# build model using VxmDense
with strategy.scope():
    vxm_model = vxm.networks.VxmDense(in_shape, nb_features, int_steps=0)


# In[ ]:


path = '../models/vxm_drosophila_2d.h5'
vxm_model.load_weights(path)
print(f'saved model in: {path}')


# validation video,

# In[ ]:


x_val = tiff.imread(orig_examples[1], key=range(200))
x_val, (x_val_low, x_val_hig) = preprocessing(x_val)

# let's get some data
val_generator = vxm_data_generator(x_val, batch_size=args.batch_size, training=False)


# In[ ]:


val_pred = []
for (val_input, _) in val_generator:
    val_pred += [vxm_model.predict(val_input, verbose=2)]
val_pred = [
    np.concatenate([a[0] for a in val_pred], axis=0),
    np.concatenate([a[1] for a in val_pred], axis=0)
]


# In[ ]:


# visualize

# In[ ]:

i, j = val_pred[1].shape[1]//2, val_pred[1].shape[2]//2
ne.plot.flow([val_pred[1][0, i:i+100, j:j+100, :].squeeze()], width=16, show=False);
plt.savefig(args.out_dir + '/example-flow.svg')

# In[ ]:


make_video(args.out_dir + '/validation-video', frame_gen(val_pred[0]))
print('generated output video')
