#!/bin/python

import os
# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import ofco
import tifffile as tiff
import numpy as np

orig_examples = [
    '../data/200901_G23xU1/Fly1/Fly1/001_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/002_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/003_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/004_coronal/2p/denoised_red.tif',
    '../data/200901_G23xU1/Fly1/Fly1/005_coronal/2p/denoised_red.tif'
]


# ## Timings

N = 2
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
