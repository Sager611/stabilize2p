"""Evaluation metrics module.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

from typing import Optional
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .utils import estimate_background_threshold, get_centers


def EMD(video: np.ndarray, n_samples: int = 100, metric: str = 'euclidean') -> float:
    """Earth Moving Distance score.
    
    This metric is not standard and it's not recomended its use, but only
    for its novelty.

    This is a generalization for the 1D Wasserstein distance.

    ``n_samples`` samples are taken for each frame, considering the frame
    an n-dimensional distribution. Then, for each two consecutive frames,
    their sampled points are run through :func:`scipy.optimize.linear_sum_assignment`.

    Parameters
    ----------
    video : array
        n-dimensional video
    n_samples : array
        number of samples to take for each frame.
        Defaults to 100.
    metric : string
        what metric to use as the distance between samples.
        Defaults to 'euclidean'.
    """
    scores = []
    
    def loop(I, J):
        # make frames probability distributions
        I = I - I.min()
        J = J - J.min()

        I = I / I.sum()
        J = J / J.sum()

        # sample points
        I_idx = np.random.choice(I.flatten().size, p=I.flatten(), size=n_samples)
        J_idx = np.random.choice(J.flatten().size, p=J.flatten(), size=n_samples)

        I_pts = np.c_[np.unravel_index(I_idx, I.shape)][:, ::-1]
        J_pts = np.c_[np.unravel_index(J_idx, J.shape)][:, ::-1]

        # calculate EMD
        d = cdist(I_pts, J_pts, metric=metric)
        assignment = linear_sum_assignment(d)
        return d[assignment].sum() / n_samples

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, I, J)
            # for each two consecutive frames
            for I, J in zip(video[:-1], video[1:])
        ]
        scores = [f.result() for f in futures]

    return np.max(scores)


def NCC(video: np.ndarray) -> float:
    """Normalized Cross-Correlation score.

    This method works on 2D and 3D video inputs.
    """
    vxm_ncc = vxm.losses.NCC()
    video = tf.convert_to_tensor(video[..., np.newaxis], dtype=np.float32)

    # vxm NCC's assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    return np.mean(vxm_ncc.loss(video[1:], video[:-1]))


def MSE(video: np.ndarray, ref: str = 'previous') -> float:
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


# def get_correlation_scores(video: np.ndarray) -> np.ndarray:
#     """For each frame, return a score from 0 to 1 on how correlated it is to the mean frame."""
#     mean_frame = video.mean(axis=0)
#     # auto-correlation.
#     # value achieved if all frames were the same
#     ref_corr = (mean_frame * mean_frame).sum()

#     def loop(i):
#         # cross-correlation
#         corr = (video[i] * mean_frame).sum()
#         # avoid numerical errors
#         large = max(ref_corr, corr)
#         local_ref_corr = ref_corr / large
#         corr /= large
#         # if corr == ref_corr we will get 1.0
#         # if corr and ref_corr are very different, we will get 0.0
#         score = 1.0 - np.abs(local_ref_corr - corr) / (local_ref_corr + corr)
#         assert abs(score) <= 1.0, f'{score=} | {local_ref_corr=} | {corr=}'
#         return score
    
#     with ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(loop, i)
#             for i in range(len(video))
#         ]
#         scores = [f.result() for f in futures]
#     scores = np.array(scores)
#     return scores


# def get_correlation_scores_prev(video: np.ndarray) -> np.ndarray:
#     """For each frame, return a score from 0 to 1 on how correlated it is to the next frame."""

#     def loop(i):
#         # auto-correlation.
#         # value achieved if all frames were the same
#         ref_corr = (video[i] * video[i]).sum()
#         # cross-correlation
#         corr = (video[i] * video[i+1]).sum()
#         # avoid numerical errors
#         large = max(ref_corr, corr)
#         local_ref_corr = ref_corr / large
#         corr /= large
#         # if corr == ref_corr we will get 1.0
#         # if corr and ref_corr are very different, we will get 0.0
#         score = 1.0 - np.abs(local_ref_corr - corr) / (local_ref_corr + corr)
#         assert abs(score) <= 1.0, f'{score=} | {local_ref_corr=} | {corr=}'
#         return score
    
#     with ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(loop, i)
#             for i in range(len(video)-1)
#         ]
#         scores = [f.result() for f in futures]
#     scores = np.array(scores)
#     return scores


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