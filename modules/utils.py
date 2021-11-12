"""Utility module containing methods used accross the project/notebooks."""

import time
import logging
import itertools
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Iterable
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import skimage.measure


def resize_shape(shape, original_shape, allow_upsampling=False):
    """
    This function converts an image shape into
    a new size respecting the ratio between
    width and height.

    Notes
    -----
    Taken from: https://github.com/NeLy-EPFL/utils_video/blob/2b18493085576b6432b3eaecd0d6d62845ea3abc/utils_video/utils.py#L25

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

    Notes
    -----
    Taken and modified from: https://github.com/NeLy-EPFL/utils_video/blob/2b18493085576b6432b3eaecd0d6d62845ea3abc/utils_video/main.py#L10

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