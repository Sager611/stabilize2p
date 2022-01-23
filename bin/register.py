#!/bin/env python
# coding: utf-8
"""General script for 2-photon imaging registration.
"""

import os
import argparse

########################### line arguments ###########################

parser = argparse.ArgumentParser()

# general parameters
_models_path_default = os.path.abspath(str(__file__) + '/../../models')
parser.add_argument('-m', '--method', required=True,
                    help='registration method. Either: pystackreg, ofco, voxelmorph, hypermorph or com+ecc')
parser.add_argument('--net', default=None,
                    help='saved network path for voxelmorph or hypermorph')
parser.add_argument('-i', '--input', nargs='+', required=True,
                    help='TIFF image(s) input paths. First provided image will be the reference to predict the deformation field,'
                         ' which will be applied to all other provided images')
parser.add_argument('-o', '--output', nargs='+', required=True,
                    help='TIFF image(s) output paths. First provided image path corresponds to the reference image,'
                         ' and the other provided paths correspond to all other provided images in --input')

# optional specific parameters
parser.add_argument('--gpu', default='',
                    help='visible GPU ID numbers. Goes into "CUDA_VISIBLE_DEVICES" env var (default: use all GPUs)')
parser.add_argument('--ref', default='first',
                    help='what image to use as reference. Used for pystackreg, voxelmorph and hypermorph. Can be: previous, first (default: first)')

args = parser.parse_args()
args.method = args.method.lower()

# validate arguments
if len(args.input) != len(args.output):
    raise ValueError(f'--input and --output must have the same length. They have lengths: in: {len(args.input)} out: {len(args.output)}')

if args.method in ['voxelmorph', 'hypermorph'] and args.net is None:
    raise ValueError(f'for method "{args.method}" you must specify a network through --net')

if args.method in ['ofco'] and len(args.input) > 2:
    raise ValueError(f'sorry! method "{args.method}" only supports one extra file to warp. You provided {len(args.input)} files')

######################################################################

# visible GPUs
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# suppress info and warn TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# general imports
import time
import logging
import numpy as np
import tifffile as tiff
import stabilize2p.register as register
from stabilize2p.utils import vxm_preprocessing, vxm_undo_preprocessing


def logstring(t1, t2, nb_frames):
    return f'Elapsed {t2-t1:3f}s | {nb_frames/(t2-t1):.4f} frames/s | {(t2-t1)/nb_frames:.4f} s/frame'

# logger
_LOGGER = logging.getLogger('stabilize2p')

_LOGGER.info(f'script arguments: {vars(args)}')

calc_warp = len(args.input) > 1

if args.method == 'pystackreg':
    from pystackreg import StackReg

    t1 = time.perf_counter()

    # load reference
    image = tiff.imread(args.input[0])
    nb_frames = image.shape[0]
    _LOGGER.info('Loaded reference %s' % args.input[0])

    # register
    sr = StackReg(StackReg.AFFINE)
    sr.register_stack(image, reference=args.ref)
    image = sr.transform_stack(image)

    t2 = time.perf_counter()
    _LOGGER.info('Calculated PyStackReg transform. %s' % logstring(t1, t2, nb_frames))

    # save
    tiff.imwrite(args.output[0], image)
    _LOGGER.info('Written image to output %s' % args.output[0])

    # free memory
    del image

    # tramsform other images
    for in_path, out_path in zip(args.input[1:], args.output[1:]):
        # load image
        image = tiff.imread(in_path)

        if image.shape[0] != nb_frames:
            _LOGGER.error(f'image "{in_path}" has {image.shape[0]} frames while {nb_frames} were expected! Skipping..')
            continue

        # warp
        image = sr.transform_stack(image)

        # save
        tiff.imwrite(out_path, image)
        _LOGGER.info('Written image %s to output %s' % (in_path, out_path))

elif args.method == 'ofco':
    import ofco

    t1 = time.perf_counter()

    # load reference
    image = tiff.imread(args.input[0])
    nb_frames = image.shape[0]
    _LOGGER.info('Loaded reference %s' % args.input[0])

    # load 2nd image
    if calc_warp:
        image2 = tiff.imread(args.input[1])
        _LOGGER.info('Loaded 2nd image %s' % args.input[1])

        if image2.shape[0] != nb_frames:
            raise ValueError(f'image "{args.input[1]}" has {image2.shape[0]} frames while {nb_frames} were expected!')
    else:
        image2 = None

    # register
    param = ofco.utils.default_parameters()
    frames = [i for i in range(nb_frames)]
    image, image2 = ofco.motion_compensate(image, image2, frames, param, verbose=True, parallel=True)

    t2 = time.perf_counter()
    _LOGGER.info('Calculated OFCO transform. %s' % logstring(t1, t2, nb_frames))

    # save
    tiff.imwrite(args.output[0], image)
    _LOGGER.info('Written image to output %s' % args.output[0])

    if image2 is not None:
        tiff.imwrite(args.output[1], image2)
        _LOGGER.info('Written image to output %s' % args.output[1])

elif args.method == 'voxelmorph':
    t1 = time.perf_counter()

    # register
    image = register.voxelmorph_transform(
        args.input[0],
        args.net,
        return_flow=calc_warp,
        data_generator_kw=dict(batch_size=128,
                               ref=args.ref)
    )
    if calc_warp:
        image, flow = image

    nb_frames = image.shape[0]

    t2 = time.perf_counter()
    _LOGGER.info('Calculated Hypermorph transform. %s' % logstring(t1, t2, nb_frames))

    # save reference
    tiff.imwrite(args.output[0], image)
    _LOGGER.info('Written image to output %s' % args.output[0])

    # free memory
    del image

    # tramsform other images
    for in_path, out_path in zip(args.input[1:], args.output[1:]):
        # load image
        image = tiff.imread(in_path)

        if image.shape[0] != nb_frames:
            _LOGGER.error(f'image "{in_path}" has {image.shape[0]} frames while {nb_frames} were expected! Skipping..')
            continue

        # warp
        inshape = image.shape[1:]
        moving = image[..., np.newaxis]
        image = vxm.networks.Transform(inshape, nb_feats=1).predict([moving, warp])
        image = image.squeeze()

        # save
        tiff.imwrite(out_path, image)
        _LOGGER.info('Written image %s to output %s' % (in_path, out_path))

elif args.method == 'hypermorph':
    t1 = time.perf_counter()

    # register
    image = register.hypermorph_optimal_transform(
        args.input[0],
        args.net,
        return_flow=calc_warp,
        data_generator_kw=dict(batch_size=128,
                               ref=args.ref)
    )
    if calc_warp:
        image, flow = image

    nb_frames = image.shape[0]

    t2 = time.perf_counter()
    _LOGGER.info('Calculated Hypermorph transform. %s' % logstring(t1, t2, nb_frames))

    # save reference
    tiff.imwrite(args.output[0], image)
    _LOGGER.info('Written image to output %s' % args.output[0])

    # free memory
    del image

    # tramsform other images
    for in_path, out_path in zip(args.input[1:], args.output[1:]):
        # load image
        image = tiff.imread(in_path)

        if image.shape[0] != nb_frames:
            _LOGGER.error(f'image "{in_path}" has {image.shape[0]} frames while {nb_frames} were expected! Skipping..')
            continue

        # warp
        inshape = image.shape[1:]
        moving = image[..., np.newaxis]
        image = vxm.networks.Transform(inshape, nb_feats=1).predict([moving, warp])
        image = image.squeeze()

        # save
        tiff.imwrite(out_path, image)
        _LOGGER.info('Written image %s to output %s' % (in_path, out_path))

elif args.method == 'com+ecc':
    from scipy import ndimage as ndi

    # load reference
    image = tiff.imread(args.input[0])
    nb_frames = image.shape[0]

    t1 = time.perf_counter()

    # apply thresholding and normalization
    image, params = vxm_preprocessing(image, affine_transform=False)

    # register
    ref = image.mean(axis=0)
    # scipy returns in (y, x) format, so we have to swap them
    target = np.array(ndi.center_of_mass(ref))[::-1]
    # best and fastest affine registration is to remove outliers with CoM
    # and then apply few ECC steps
    image = register.com_transform(image, target=target, return_transform=calc_warp)
    if calc_warp:
        image, shifts = image
    image = register.ECC_transform(image, ref=ref, nb_iters=3, return_transform=calc_warp)
    if calc_warp:
        image, warps = image

    t2 = time.perf_counter()
    _LOGGER.info('Calculated COM-T+ECC transform. %s' % logstring(t1, t2, nb_frames))

    # save reference image
    image = vxm_undo_preprocessing(image, params)
    tiff.imwrite(args.output[0], image)
    _LOGGER.info('Written image to output %s' % args.output[0])

    # free memory
    del image

    # transform other images
    for in_path, out_path in zip(args.input[1:], args.output[1:]):
        # load image
        image = tiff.imread(in_path)

        if image.shape[0] != nb_frames:
            _LOGGER.error(f'image "{in_path}" has {image.shape[0]} frames while {nb_frames} were expected! Skipping..')
            continue

        # shift
        image = register.com_transform(image, target=target, shifts=shifts)
        
        # warp
        image = register.ECC_transform(image, warps=warps)

        # save
        tiff.imwrite(out_path, image)
        _LOGGER.info('Written image %s to output %s' % (in_path, out_path))

else:
    raise ValueError(f'--method must be one of: pystackreg, ofco, voxelmorph, hypermorph or com+ecc. Provided: {args.method}')

_LOGGER.info('DONE')