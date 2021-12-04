#!/usr/bin/env python
# coding: utf-8

# # Optical Flow exploration notebook

# In[1]:


import os
# both for tensorflow and pycuda
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import tifffile as tiff
import numpy as np
import matplotlib
import voxelmorph as vxm
import tensorflow as tf
import nibabel as nib
import neurite as ne
from matplotlib import pyplot as plt

# Font size
plt.style.use('default')

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[4]:


from stabilize2p.utils import make_video

def frame_gen(video, scores=None, lt=0.9):
    low, hig = video[0].min(), video[1].max()
    if scores is not None:
        for img, score in zip(video, scores):
            img = (img - low) / (hig - low) * 255
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
            img = (img - low) / (hig - low) * 255
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype(np.uint8)
            yield img


# In[5]:


orig_examples = [
    '../data/200901_G23xU1/Fly1/Fly1/001_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/002_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/003_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/004_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/005_coronal/2p/denoised_red.tif'
]


# ## Timings

# first, we check the git branches in the ofco source code,

# In[1]:


get_ipython().system('cd ../src/ofco && git branch')


# ### NUMPY with process-level parallelism
# 
# <pre>
# results (20 frames):                       <mono>Elapsed 265.1s | 13.26s per frame</mono>
# with thread-level parallelism (20 frames): <mono>Elapsed 227.7s | 11.39s per frame</mono>
# </pre>
# 
# This is the old -> slower approach.

# In[5]:


# first we make sure we are in the main branch
get_ipython().system(' cd ../src/ofco && git checkout master')

import time
import logging
from ofco import motion_compensate
from ofco.utils import default_parameters

# by defualt Jupyter does not show info logs, so we activate them
logging.getLogger().setLevel(logging.INFO)

# number of frames to register
N = 20

key = range(N)

stack1 = tiff.imread(orig_examples[1], key=key)

param = default_parameters()
frames = [i for i in key]

t1 = time.perf_counter()
stack1_warped, _ = motion_compensate(stack1, None, frames, param,
                                     verbose=True, parallel=True)
t2 = time.perf_counter()
print(f'Elapsed {t2-t1:.4g}s | {N/(t2-t1):,.0f} frames/s | {(t2-t1)/N:.4g} s/frame')


# In[6]:


make_video('ofco-master', frame_gen(stack1_warped))
numpy_ofco = stack1_warped


# ### TENSORFLOW with thread-level parallelism
# 
# `Elapsed 169.9s | 8.494s per frame`
# 
# Here we replaced fft2 and ifft2 with the tensorflow's functions

# In[5]:


# first we make sure we are in the feature branch
get_ipython().system(' cd ../src/ofco && git checkout fft-tf')

import time
from ofco import motion_compensate
from ofco.utils import default_parameters

N = 20

key = range(N)

stack1 = tiff.imread(orig_examples[1], key=key)

param = default_parameters()
frames = [i for i in key]

t1 = time.perf_counter()
stack1_warped, _ = motion_compensate(stack1, None, frames, param,
                                     verbose=True, parallel=True)
t2 = time.perf_counter()
print(f'Elapsed {t2-t1:.4g}s | {N/(t2-t1):,.0f} frames/s | {(t2-t1)/N:.4g} s/frame')


# In[6]:


make_video('ofco-fft-tf', frame_gen(stack1_warped))

if not np.allclose(stack1_warped, numpy_ofco):
    print('WARNING: NOT THE SAME')