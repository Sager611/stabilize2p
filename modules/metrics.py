"""Evaluation metrics module."""

from typing import Optional

import numpy as np
from scipy import ndimage
from concurrent.futures.thread import ThreadPoolExecutor


def get_centers(video: np.ndarray) -> np.ndarray:
    """Return each frame's center of mass."""
    # use parallelism
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(ndimage.center_of_mass, frame)
            for frame in video
        ]
        centers = [f.result() for f in futures]
    centers = np.stack(centers)
    return centers


def failure_score(video: np.ndarray, frame_shape: Optional[tuple] = None) -> np.ndarray:
    """Return percentage of frames considered 'failed' due to remoteness to the center.

    An important assumption this score makes is that the mean over the centers of mass of
    the input :param:`video` is suposed to represent a proper center were all axons are
    visible.
    """
    if frame_shape is None:
        frame_shape = (video.shape[1], video.shape[2])

    centers = get_centers(video)

    # consider a frame wrong if the axis is > 1% off the mean
    off_x = 0.01 * frame_shape[0]
    off_y = 0.01 * frame_shape[1]
    m_center = centers.mean(axis=0)
    failures = np.sum((np.abs(centers[:, 0] - m_center[0]) > off_x) | (np.abs(centers[:, 1] - m_center[1]) > off_y))
    return failures / centers.shape[0], failures