"""Evaluation metrics module.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

from typing import Optional

import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error

from .utils import estimate_background_threshold, get_centers


def MSE_score(video: np.ndarray, ref: str = 'previous') -> float:
    """Return MSE score with respect to some reference image.
    
    This method is tested on 2D videos, but should work on 3D videos as well.

    Parameters
    ----------
    video : array
        Contains the video information
    ref : string
        Reference frame/image to use as approx for round-truth.
        Either: previous, median or mean
        Default is 'previous'
    """
    nb_frame_pixels = np.prod(video.shape[1:])
    
    # pre-processing
    video = (video - video.mean())
    # video = (video - video.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]) / (1e-6 + video.std(axis=(1, 2))[:, np.newaxis, np.newaxis])
    
    if ref == 'previous':
        I = video[:-1].reshape((-1, nb_frame_pixels))
    elif ref == 'median':
        I = np.tile(np.median(video, axis=0).ravel(), (video.shape[0], 1))
    elif ref == 'mean':
        I = np.tile(np.mean(video, axis=0).ravel(), (video.shape[0], 1))
    else:
        raise ValueError(f'Reference "{ref}" is not recognized. Recognized references: previous, median, mean')
    
    if ref == 'previous':
        J = video[1:].reshape((-1, nb_frame_pixels))
    else:
        J = video.reshape((-1, nb_frame_pixels))
    
    return mean_squared_error(I, J, multioutput='uniform_average')


def failure_score(video: np.ndarray,
                  frame_shape: Optional[tuple] = None,
                  threshold: Optional[int] = None) -> np.ndarray:
    """Return percentage of frames considered 'failed' due to remoteness to the center.

    An important assumption this score makes is that the mean over the centers of mass of
    the input ``video`` is suposed to represent a proper center were all axons are
    visible.
    """
    if frame_shape is None:
        frame_shape = (video.shape[1], video.shape[2])
    if threshold is None:
        threshold = 0.01 * min(*frame_shape)

    # pre-processing
    bkg = estimate_background_threshold(video[0])
    # makes a copy of video, so we don't modify the argument
    video = np.clip(video, bkg, None)
    video = video - bkg

    centers = get_centers(video)

    # consider a frame wrong if the axis is > 10% off the mean
    m_center = centers.mean(axis=0)
    failures = np.sum((centers - m_center)**2, axis=1) > threshold**2
    return np.sum(failures) / centers.shape[0], failures


def get_correlation_scores(video: np.ndarray) -> np.ndarray:
    """For each frame, return a score from 0 to 1 on how correlated it is to the mean frame."""
    mean_frame = video.mean(axis=0)
    # auto-correlation.
    # value achieved if all frames were the same
    ref_corr = (mean_frame * mean_frame).sum()

    def loop(i):
        # cross-correlation
        corr = (video[i] * mean_frame).sum()
        # avoid numerical errors
        large = max(ref_corr, corr)
        local_ref_corr = ref_corr / large
        corr /= large
        # if corr == ref_corr we will get 1.0
        # if corr and ref_corr are very different, we will get 0.0
        score = 1.0 - np.abs(local_ref_corr - corr) / (local_ref_corr + corr)
        assert abs(score) <= 1.0, f'{score=} | {local_ref_corr=} | {corr=}'
        return score
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, i)
            for i in range(len(video))
        ]
        scores = [f.result() for f in futures]
    scores = np.array(scores)
    return scores


def get_correlation_scores_prev(video: np.ndarray) -> np.ndarray:
    """For each frame, return a score from 0 to 1 on how correlated it is to the next frame."""

    def loop(i):
        # auto-correlation.
        # value achieved if all frames were the same
        ref_corr = (video[i] * video[i]).sum()
        # cross-correlation
        corr = (video[i] * video[i+1]).sum()
        # avoid numerical errors
        large = max(ref_corr, corr)
        local_ref_corr = ref_corr / large
        corr /= large
        # if corr == ref_corr we will get 1.0
        # if corr and ref_corr are very different, we will get 0.0
        score = 1.0 - np.abs(local_ref_corr - corr) / (local_ref_corr + corr)
        assert abs(score) <= 1.0, f'{score=} | {local_ref_corr=} | {corr=}'
        return score
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, i)
            for i in range(len(video)-1)
        ]
        scores = [f.result() for f in futures]
    scores = np.array(scores)
    return scores


def cont_dice_scores(video: np.ndarray) -> float:
    normals = np.sum(video ** 2, axis=(1, 2))
    N = normals.max()

    def loop(i):
        return np.sum(video[i] * video[i+1]) / N

    # use parallelism
    res = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, i)
            for i in range(len(video)-1)
        ]
        res = [f.result() for f in futures]
    res = np.array(res)
    return res