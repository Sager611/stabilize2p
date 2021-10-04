import itertools
import subprocess

import cv2
from tqdm import tqdm
from typing import Iterable

import numpy as np


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


def make_video(video_path: str, frame_generator: Iterable, fps: int = 8, output_shape: tuple[int, int] = (-1, 2880), n_frames: int = -1):
    """
    This function writes a video to file with all frames that
    the `frame_generator` yields.

    Notes
    -----
    Taken from: https://github.com/NeLy-EPFL/utils_video/blob/2b18493085576b6432b3eaecd0d6d62845ea3abc/utils_video/main.py#L10

    Parameters
    ----------
    video_path : string
        Name/path to the output file.
    frame_generator : generator
        Generator yielding individual frames.
    fps : int
        Frame rate in frames per second.
    """
    if float(fps).is_integer() and int(fps) != 1 and (int(fps) & (int(fps) - 1)) == 0:
        import warnings

        warnings.warn(
            f"Frame rate {fps} is a power of 2. This can result in faulty video files."
        )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first_frame = next(frame_generator)
    frame_generator = itertools.chain([first_frame], frame_generator)
    output_shape = resize_shape(output_shape, first_frame.shape[:2])
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, output_shape[::-1])

    for frame, img in tqdm(enumerate(frame_generator)):
        resized = cv2.resize(img, output_shape[::-1])
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb)
        if frame == n_frames - 1:
            break

    video_writer.release()