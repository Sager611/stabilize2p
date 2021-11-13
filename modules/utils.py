"""Utility module containing methods used accross the project/notebooks.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

import os
import math
import time
import itertools
import logging
import itertools
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Iterable
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np
import skimage.measure
import tifffile as tiff
import tensorflow as tf
import voxelmorph as vxm
from matplotlib import pyplot as plt


def largest_divisor_lte(n, lim):
    """Returns the largest divisor of :param:`n` that is less-than or equal to :param:`lim`."""
    large_divisors = []
    last_div = 1
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            if i > lim:
                return last_div
            last_div = i
            if i*i != n:
                large_divisors.append(n // i)
    for divisor in reversed(large_divisors):
        if divisor > lim:
            return last_div
        last_div = divisor
    return n


def split_video(path: str, W: int, H: int, target_shape: tuple, stepx: int, stepy: int, batch_size: int = 5, key=None):
    """Given the path to a tiff file, generate sliding windows of shape :param:`target_shape` for each frame.
    
    This method is part of :func:`register`.
    
    :param key: specify the frames to analyze.
        For example: ``key = range(50, 100, 2)``, ``key = [1, 2, 3]``, etc.
        Defaults to all use all frames.
    """
    if key is not None:
        if type(key) is range:
            key = [i for i in key]
        key = set(key)

    with tiff.TiffFile(path) as tif:
        batch, pos_array, norm_array = [], [], []
        for fi, page in enumerate(tif.pages):

            # skip unwanted frames
            if key is not None and fi not in key:
                continue

            logging.info(f'frame: {fi}')
            frame = page.asarray()
            for i, j in itertools.product(range(0, H-target_shape[0] +1, stepy), range(0, W-target_shape[1] +1, stepx)):
                # col == x ; row == y
                sub_img = frame[i:i+target_shape[0], j:j+target_shape[1]]
                
                # normalize to [0, 1]
                low, hig = sub_img.min(), sub_img.max()
                sub_img = (sub_img - low) / (hig - low + 1e-6)
                
                batch += [sub_img]
                pos_array += [[fi, i, j]]
                norm_array += [[low, hig]]
                if len(batch) == batch_size:
                    batch = np.stack(batch)

                    # if the expected input is 3D, simply tile the array
                    if len(target_shape) > 2:
                        for _ in range(len(target_shape)-2):
                            batch = batch[..., np.newaxis]
                        tile_shape = [1] * len(batch.shape)
                        tile_shape[3] = target_shape[2]
                        batch = np.tile(batch, tile_shape)

                    yield batch, np.array(pos_array), np.array(norm_array)
                    batch = []
                    pos_array.clear()
                    norm_array.clear()
        
    # last batch
    if len(batch) > 0:
        batch = np.stack(batch)

        # if the expected input is 3D, simply tile the array
        if len(target_shape) > 2:
            for _ in range(len(target_shape)-2):
                batch = batch[..., np.newaxis]
            tile_shape = [1] * len(batch.shape)
            tile_shape[3] = target_shape[2]
            batch = np.tile(batch, tile_shape)

        yield batch, np.array(pos_array), np.array(norm_array)


def reconstruct_video(moved_subimgs: np.ndarray,
                      flow_subimgs: np.ndarray,
                      pos_arrays: np.ndarray,
                      norm_arrays: np.ndarray,
                      W: int, H: int, stepx: int, stepy: int) -> tuple[np.ndarray, np.ndarray]:
    """Given the splitting of :func:`split_video`, reconstruct it.
    
    This method is part of :func:`register`.
    
    .. warning::
        At the moment flow is an empty array.
    
    TODO: reconstruct flow!!
    """
    nb_frames = pos_arrays[-1, 0] + 1
    out_video = np.empty((nb_frames, H, W))
    out_flow = np.empty((nb_frames, *flow_subimgs.shape[1:]))
    subimg_R, subimg_C = moved_subimgs.shape[1:]
    # subflow_R, subflow_C = flow_subimgs.shape[1:]

    nb_x_subimgs = 1 + (W - subimg_C) // stepx
    nb_y_subimgs = 1 + (H - subimg_R) // stepy

    marginx = (subimg_C - stepx) // 2
    marginy = (subimg_R - stepy) // 2
    logging.info(f'{marginx=} | {marginy=}')

    for (fi, i, j) in pos_arrays:
        si = (i) // stepy
        sj = (j) // stepx
        subimg_i = fi*nb_x_subimgs*nb_y_subimgs + si*nb_x_subimgs + sj
        
        off_left, off_up = marginx, marginy
        if j == 0:
            off_left = 0
        if i == 0:
            off_up = 0

        idx = (fi, slice(i +off_up, i+subimg_R), slice(j +off_left, j+subimg_C))
        
        out_video[idx] = moved_subimgs[subimg_i, off_up:, off_left:]

        # unnormalize
        low, hig = norm_arrays[subimg_i]
        out_video[idx] = out_video[idx] * (hig - low) + low
        
        # out_video[idx] = 1 if (si+sj) % 2 == 0 else 0

        # out_flow[fi, i:i+subflow_R, j:j+subflow_C, :] = flow_subimgs[fi]
    return out_video, out_flow


def vxm_register(video_path: str, model_weights_path: str, batch_size: int = 5, strategy: str = 'default', key=None) -> tuple[np.ndarray, np.ndarray]:
    """Given a TIFF video path, and a voxelmorph model, stabilize the video.
    
    .. warning:: 
        At the moment flow is an empty array.
    
    :param video_path: full-path to the TIFF file.
    :param model_weights_path: full-path to the `h5` file with the weights for the Voxelmorph model to be used.
    :param batch_size: size of the batch.
    :param strategy: either: 'default', 'GPU' (uses all GPUs), 'TPU', 'GPU:0', 'GPU:1', ...
        Defaults to 'default', which uses the CPU.
    :param key: specify the frames of :param:`video_path` to analyze.
        For example: ``key = range(50, 100, 2)``, ``key = [1, 2, 3]``, etc.
        Defaults to all use all frames.
    """
    # decrease logging level for tensorflow
    PREV_TF_LOGGING_LEVEL = logging.getLogger().level
    tf.get_logger().setLevel('FATAL')
    
    # increase logging level for our code
    PREV_LOGGING_LEVEL = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)

    # TPU or GPU or CPU
    strategy = get_strategy(strategy)
    
    # load model
    t1 = time.perf_counter()
    with strategy.scope():
        vxm_model = vxm.networks.VxmDense.load(model_weights_path, input_model=None)
    t2 = time.perf_counter()
    logging.info(f'Loaded model in {t2-t1:.2g}s')
    
    # print model's layers and info
    vxm_model.summary(line_length = 180)
    
    target_shape = tuple(vxm_model.input[0].shape[1:])
    logging.info(f'{target_shape=}')

    # get width/height info
    img = tiff.imread(video_path, key=[0])
    if len(img.shape) != 2:
        raise ValueError(f'Only frame formats supported are of the shape (W x H). Provided frame shape: {img.shape}')
    H, W = img.shape
    del img
    
    # setup window sliding parameters
    stepx = largest_divisor_lte(W - target_shape[1], target_shape[1])
    stepy = largest_divisor_lte(H - target_shape[0], target_shape[0])
    
    moved_subimgs = []
    flow_subimgs = []
    pos_arrays = []
    norm_arrays = []

    t1 = time.perf_counter()
    for batch, pos_array, norm_array in split_video(video_path, W, H, target_shape, stepx, stepy, batch_size=batch_size, key=key):
        moved, flow = vxm_model.predict([batch, batch])

        logging.info(f'{moved.shape=} | {flow.shape=}')

        moved_subimgs += [moved[..., 0, 0]]
        flow_subimgs += [flow[..., 0, :]]
        pos_arrays += [pos_array]
        norm_arrays += [norm_array]
    t2 = time.perf_counter()

    moved_subimgs = np.concatenate(moved_subimgs)
    flow_subimgs = np.concatenate(flow_subimgs)
    pos_arrays = np.concatenate(pos_arrays)
    norm_arrays = np.concatenate(norm_arrays)

    nb_frames = pos_arrays[-1, 0] + 1
    logging.info(f'Rate: {(t2-t1)/nb_frames : .4g} s / frame')

    out_video, out_flow = reconstruct_video(moved_subimgs, flow_subimgs, pos_arrays, norm_arrays, W, H, stepx, stepy)
    logging.info(f'{out_video.shape=} | {out_flow.shape=}')

    # return the logging levels as they where before
    tf.get_logger().setLevel(PREV_TF_LOGGING_LEVEL)
    logging.getLogger().setLevel(PREV_LOGGING_LEVEL)

    return out_video, out_flow

def resize_shape(shape, original_shape, allow_upsampling=False):
    """
    This function converts an image shape into
    a new size respecting the ratio between
    width and height.

    .. note::
        Taken from: `here <https://github.com/NeLy-EPFL/utils_video/blob/2b18493085576b6432b3eaecd0d6d62845ea3abc/utils_video/utils.py#L25/>`_

    Parameters
    ----------
    shape : tuple of two integers
        Desired shape. The tuple and contain one, two
        or no -1 entry. If no entry is -1, this argument
        is returned. If both entries are -1, `original_shape`
        is returned. If only one of the entires is -1, its new
        value in `new_shape` is calculated preserving the ratio
        of `original_shape`.
    original_shape : tuple of two integers
        Original shape.
    Returns
    -------
    new_shape : tuple of two integers
        Resized shape.
    """
    if len(shape) != 2:
        raise ValueError("shape has to be of length 2.")
    if len(original_shape) != 2:
        raise ValueError("original_shape has to be of length 2.")
    if shape[0] % 1 != 0 or shape[1] % 1 != 0:
        raise ValueError("Entries of shape have to be integers.")
    if original_shape[0] % 1 != 0 or original_shape[1] % 1 != 0:
        raise ValueError("Entries of original_shape have to be integers.")
    if np.any(np.array(shape) < -1):
        raise ValueError("The values of shape cannot be smaller than -1.")
    if np.any(np.array(original_shape) < -1):
        raise ValueError("The values of original_shape cannot be smaller than -1.")

    if shape[0] == -1 and shape[1] == -1:
        new_shape = original_shape
    elif shape[0] == -1:
        ratio = original_shape[0] / original_shape[1]
        new_shape = (int(shape[1] * ratio), shape[1])
    elif shape[1] == -1:
        ratio = original_shape[1] / original_shape[0]
        new_shape = (shape[0], int(shape[0] * ratio))
    else:
        new_shape = shape
    if not allow_upsampling:
        if new_shape[0] > original_shape[0] and new_shape[1] > original_shape[1]:
            return original_shape
    return new_shape


def make_video(video_path: str,
               frame_generator: Iterable,
               fps: int = 9,
               cmap=cv2.COLORMAP_VIRIDIS,
               ext: str = 'mp4',
               output_shape: tuple[int, int] = (-1, 2880),
               n_frames: int = -1,
               output_format: str = 'mov'):
    """This function writes a video to file with all frames that the :param:`frame_generator` yields.

    .. note::
        Taken and modified from: `here <https://github.com/NeLy-EPFL/utils_video/blob/2b18493085576b6432b3eaecd0d6d62845ea3abc/utils_video/main.py#L10/>`_

    :param video_path: Name/path to the output file.
    :param frame_generator: Generator yielding individual frames.
    :param fps: Frame rate in frames per second.
    :param output_format: format of the output video. Defauts to `"mov"`.
    """
    if float(fps).is_integer() and int(fps) != 1 and (int(fps) & (int(fps) - 1)) == 0:
        import warnings

        warnings.warn(
            f"Frame rate {fps} is a power of 2. This can result in faulty video files."
        )

    if video_path.split('.')[-1] in ['mov', 'mp4', 'avi']:
        raise ValueError('Please do not provide an extension to the output video file')

    # if ext == 'mov':
    #     fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    if ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        raise ValueError(f'Invalid extension: {ext}')

    if output_format == "mp4":
        tmp_video_path = video_path + '.tmp.' + ext
    else:
        tmp_video_path = video_path + '.' + ext

    first_frame = next(frame_generator)
    frame_generator = itertools.chain([first_frame], frame_generator)
    output_shape = resize_shape(output_shape, first_frame.shape[:2])
    video_writer = cv2.VideoWriter(tmp_video_path, fourcc, fps, output_shape[::-1])

    for frame, img in tqdm(enumerate(frame_generator)):
        # img_out = cv2.resize(img, output_shape[::-1])
        if cmap is not None and (len(img.shape) <= 2 or img.shape[2] == 1):
            img_out = cv2.applyColorMap(img, cmap)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        else:
            img_out = img
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        video_writer.write(img_out)
        if frame == n_frames - 1:
            break

    video_writer.release()

    # for some reason, we have to convert using ffmpeg to .mov
    # otherwise we cannot see the video in HTML5 in JupyterLab
    print('converting video..')
    t1 = time.perf_counter()
    conversion_path = video_path + f".{output_format}"
    subprocess.run(["ffmpeg", "-y", "-v", "warning", "-i", tmp_video_path, "-f", output_format, conversion_path])
    t2 = time.perf_counter()
    print(f'Done ({t2-t1:.2f}s)')

    # delete old file
    Path(tmp_video_path).unlink()


def clip_video(video: np.ndarray, q1: int = 0.05, q2: int = 0.95) -> None:
    """Clip :param:`video` in-place according to lower-quantile :param:`q1` and higher-quantile :param:`q2`."""
    low_bound = np.quantile(video, q1)
    hig_bound = np.quantile(video, q2)

    # clip in-place
    np.clip(video, low_bound, hig_bound, out=video)


def segment_video(video: np.ndarray,
                  sigma: float = 1.0,
                  num_reps: int = 20,
                  contrast: float = 512.0,
                  apply_threshold: bool = True,
                  *args, **kwargs) -> np.ndarray:
    """Apply a gaussian filter to :param:`video` and use Otsu thresholding.

    :param sigma: sigma value to use for the gaussian filter.
    :param num_reps: how many times to repeat the gaussian filter for each frame.
    :param contrast: how much bring the values of the image closer to the mean. Note that higher
        values imply lower contrast, and vice-versa.
        The following formula is applied to each pixel of each frame:
        ```
            pixel <- (pixel / mean) ^ (1 / contrast) * mean
        ```
        Where `mean` is the mean over the pixels of the frame.
        If `contrast=1`, then there is no change.
    :param apply_threshold: if `False`, perform the pre-processing steps, without applying Otsu
        thresholding. It will also not convert the frames to uint8.
        Defaults to `True`.
    :param *args: extra positional arguments for `cv2.GaussianBlur`.
    :param **kwargs: extra keyword arguments for `cv2.GaussianBlur`.
    """
    # applying a gaussian filter repeatedly is equivalent to this single
    # gaussian filter
    sigma = sigma * np.sqrt(num_reps)

    def loop(frame):
        # reduce contrast
        mean = frame.mean()
        frame = np.power(frame / mean, 1/contrast) * mean
        # to uint8
        if apply_threshold:
            low = np.quantile(frame, q=0.01)
            std = frame.std()
            frame = np.clip(((frame-low) / std * 32), 0, 255)
            frame = frame.astype(np.uint8)

        if sigma > 0.0:
            frame = cv2.GaussianBlur(frame, None, sigma, *args, **kwargs)

        # apply Otsu thresholding
        if apply_threshold:
            _, frame = cv2.threshold(frame, 0, 1, cv2.THRESH_OTSU)
        return frame

    # use parallelism
    res = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, frame)
            for frame in video
        ]
        res = [f.result() for f in futures]
    return np.stack(res)


def blur_video(video: np.ndarray,
               ksize: tuple[int, int] = (15, 15),
               sigmaX: float = 0.0,
               num_reps: int = 1,
               *args, **kwargs) -> np.ndarray:
    """Apply a gaussian filter to :param:`video`."""
    def loop(frame):
        for _ in range(num_reps):
            frame = cv2.GaussianBlur(frame, ksize, sigmaX, *args, **kwargs)
        return frame

    # use parallelism
    res = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, frame)
            for frame in video
        ]
        res = [f.result() for f in futures]
    return np.stack(res)


def plot_frame_values_3d(frame: np.ndarray, title: str = r"Frame values", pool: int = 15, size: int = 3, saveto=None) -> None:
    """Generate 3d bar plot of :param:`frame`.

    :param frame: input frame matrix
    :param title: title for the plot
    :param pool: size of the averaging pool
    :param size: length of the bars in the x and y axes. Defaults to `1`.
    :param saveto: path to save the plot to. Defaults to `None`.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(25, 110)
    
    dz = skimage.measure.block_reduce(frame, (pool, pool), np.mean)
    
    _x, _y = np.arange(dz.shape[1]), np.arange(dz.shape[0])
    xx, yy = np.meshgrid(_x, _y)
    x, y = xx.ravel() * pool, yy.ravel() * pool
    
    dz = dz.ravel()
    
    cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz]
    
    ax.bar3d(x, y, np.zeros_like(dz), size, size, dz, color=rgba, zsort='average', shade=True)
    plt.title(title)
    plt.xlabel(r"X")
    plt.ylabel(r"Y")
    if saveto:
        plt.savefig(saveto)
    plt.show()


def get_strategy(strategy: str = 'default'):
    """Load and return the specified Tensorflow's strategy.
    
    :param strategy: either:
        * 'default' (CPU, defaults to this)
        * 'GPU'
        * 'TPU'
        * 'GPU:<gpu-index>':
            If :param:`strategy` is 'GPU', then use all GPUs.
            If :param:`strategy` is 'GPU:0', then use GPU 0.
            If :param:`strategy` is 'GPU:1', then use GPU 1.
            etc.
    """
    # print device info
    logging.info(f"Num Physical GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    logging.info(f"Num Logical  GPUs Available: {len(tf.config.list_logical_devices('GPU'))}")
    logging.info(f"Num TPUs Available: {len(tf.config.list_logical_devices('TPU'))}")

    if not tf.test.is_built_with_cuda():
        logging.warning('Tensorflow is not built with GPU support!')

    # try to allow growth in case other people are using the GPUs
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            logging.warning(f'GPU device "{gpu}" is already initialized.')

    # choose strategy
    if strategy.lower() == 'tpu' and tf.config.list_physical_devices('TPU'):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.TPUStrategy(resolver)
        logging.info(r'using TPU strategy.')
    if strategy.lower()[:3] == 'gpu' and tf.config.list_physical_devices('GPU'):
        if len(strategy) > 3:
            strategy = tf.distribute.MirroredStrategy([strategy])
        else:
            strategy = tf.distribute.MirroredStrategy()
        logging.info(r'using GPU "MirroredStrategy" strategy.')
    else:
        # use default strategy
        strategy = tf.distribute.get_strategy()
        logging.info(r'using default strategy.')

    return strategy