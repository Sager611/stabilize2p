#!/usr/bin/env python

import time
import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--batch', action='store_true',
                    help='specify that data is a batch of samples')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
add_batch_axis = not args.batch
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=add_batch_axis, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=add_batch_axis, add_feat_axis=add_feat_axis, ret_affine=True)

print(f'{moving.shape=}')
print(f'{fixed.shape=}')

inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1]

with tf.device(device):
    # load model and predict
    print('LOADING MODEL...')
    t1 = time.perf_counter()
    vxm_model = vxm.networks.VxmDense.load(args.model, input_model=None)
    t2 = time.perf_counter()
    print('-'*32)
    print(f'Loaded model in {t2-t1:.2f}s')
    print('-'*32)

    t1 = time.perf_counter()
    warp = []
    for f_moving, f_fixed in zip(moving, fixed):
        f_moving = f_moving[np.newaxis, ...]
        f_fixed = f_fixed[np.newaxis, ...]
        warp += [vxm_model.register(f_moving, f_fixed)]
    del vxm_model
    warp = np.stack(warp).squeeze()
    t2 = time.perf_counter()
    print('-'*32)
    print(f'Calculated warp in {t2-t1:.2f}s | {warp.shape=}')
    print('-'*32)

    t1 = time.perf_counter()
    moved = []
    for f_moving, f_warp in zip(moving, warp):
        f_moving = f_moving[np.newaxis, ...]
        f_warp = f_warp[np.newaxis, ...]
        moved += [vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([f_moving, f_warp])]
    t2 = time.perf_counter()
    moved = np.stack(moved)
    print('-'*32)
    print(f'Registered image in {t2-t1:.2f}s | {moving.shape[0]/(t2-t1):.2g} frame / s')
    print('-'*32)

# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)
