#!/usr/bin/env sh
# coding: utf-8

python -c "
import os
import pickle
os.makedirs('ofco-test.out', exist_ok=True)
timings = {'master': [], 'fft-tf': []}
with open('ofco-test.out/timings.pkl', 'wb') as f:
    pickle.dump(timings, f)
"

echo "" > ../logs/ofco-test.log

for N in 2 16 128 1024; do
    cd ../src/ofco && git checkout master
    cd ../../bin
    command="
import os
# both for tensorflow and pycuda
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import importlib
import logging
import pickle
import time
import sys

import cv2
import ofco
import tifffile as tiff
import numpy as np
import matplotlib
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


# ### NUMPY with process-level parallelism
# 
# <pre>
# results (20 frames):                       <mono>Elapsed 265.1s | 13.26s per frame</mono>
# with thread-level parallelism (20 frames): <mono>Elapsed 227.7s | 11.39s per frame</mono>
# </pre>
# 
# This is the old -> slower approach.

# In[5]:


with open('ofco-test.out/timings.pkl', 'rb') as f:
    timings = pickle.load(f)


N = $N
print('-'*70)
print(f'N: {N}')
print('-'*70)

key = range(N)

stack1 = tiff.imread(orig_examples[1], key=key)

param = ofco.utils.default_parameters()
frames = [i for i in key]

t1 = time.perf_counter()
stack1_warped, _ = ofco.motion_compensate(stack1, None, frames, param, parallel=True)
t2 = time.perf_counter()
print(f'Elapsed master {t2-t1:.4g}s | {N/(t2-t1):,.0f} frames/s | {(t2-t1)/N:.4g} s/frame')

timings['master'] += [(t2-t1)/N]

with open('ofco-test.out/timings.pkl', 'wb') as f:
    pickle.dump(timings, f)
    print(timings)

np.save('ofco-test.out/master', stack1_warped)

# In[6]:

# make_video('ofco-fft-tf', frame_gen(stack1_warped))
    "
    echo "MASTER"
    python -c "$command" 2>&1 | tee -a ../logs/ofco-test.log

    cd ../src/ofco && git checkout fft-tf
    cd ../../bin
    command="
import os
# both for tensorflow and pycuda
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import importlib
import logging
import pickle
import time
import sys

import cv2
import ofco
import tifffile as tiff
import numpy as np
import matplotlib
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

# In[5]:


with open('ofco-test.out/timings.pkl', 'rb') as f:
    timings = pickle.load(f)


N = $N
print('-'*70)
print(f'N: {N}')
print('-'*70)

# ### TENSORFLOW with thread-level parallelism
# 
# Elapsed 169.9s | 8.494s per frame
# 
# Here we replaced fft2 and ifft2 with the tensorflow's functions

# In[5]:

key = range(N)

stack1 = tiff.imread(orig_examples[1], key=key)

param = ofco.utils.default_parameters()
frames = [i for i in key]

t1 = time.perf_counter()
stack1_warped, _ = ofco.motion_compensate(stack1, None, frames, param, verbose=True, parallel=True)
t2 = time.perf_counter()
print(f'Elapsed fft-tf {t2-t1:.4g}s | {N/(t2-t1):,.0f} frames/s | {(t2-t1)/N:.4g} s/frame')

timings['fft-tf'] += [(t2-t1)/N]

with open('ofco-test.out/timings.pkl', 'wb') as f:
    pickle.dump(timings, f)
    print(timings)

np.save('ofco-test.out/fft-tf', stack1_warped)

# make_video('ofco-fft-tf', frame_gen(stack1_warped))
"
    echo "FFT-TF"
    python -c "$command" 2>&1 | tee -a ../logs/ofco-test.log
    
    # compare
    python -c "
import numpy as np
a = np.load('ofco-test.out/fft-tf.npy')
b = np.load('ofco-test.out/master.npy')
res = np.allclose(a, b)
print(f'{a.shape=} | {b.shape=}')
print(f'{a.mean()=} | {b.mean()=}')
print('ALLCLOSE?:', res)
print('RMSE:     ', np.sqrt(np.mean((b - a) ** 2)))
" 2>&1 | tee -a ../logs/ofco-test.log
done