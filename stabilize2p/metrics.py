"""Evaluation metrics module.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

from typing import Optional, Union
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import gen_batches

from .utils import get_centers


def EMD(video: np.ndarray, n_samples: int = 50, metric: str = 'euclidean', feat_frac: float = 0.2) -> float:
    """Earth Moving Distance score (loosely inspired).
    
    This metric is not standard and it's not recomended to use practically, but only
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
        distance metric to use for optimal transport.
        Defaults to 'euclidean'
    feat_frac : float
        the amount, fraction, of the image that corresponds to the features.
        In a formula:
        
        ``feat_frac = (# feature pixels) / (# total pixels)``
        
        Defaults to 0.1
    """
    scores = []
    
    def loop(I, J):
        # make frames probability distributions
        I = I - I.min()
        J = J - J.min()

        I = I / I.sum()
        J = J / J.sum()

        I_flat = I.flatten()
        J_flat = J.flatten()

        # first, we choose some candidates
        N = int(feat_frac * I_flat.size)
        I_idx = np.argpartition(I_flat, -N)[-N:]
        J_idx = np.argpartition(J_flat, -N)[-N:]

        # now, we sample evenly among the candidates
        I_pts = I_idx[::(I_idx.size//n_samples)]
        J_pts = J_idx[::(J_idx.size//n_samples)]
        I_pts = np.c_[np.unravel_index(I_pts, I.shape)][:, ::-1]
        J_pts = np.c_[np.unravel_index(J_pts, J.shape)][:, ::-1]

        # # not really a distance metric, since it does not
        # # follow the triangle inequality
        # W = (I + J)/2  # W in [0, 1]
        # W = W.T
        # metric = lambda u, v: (((u - v) * (1+W[tuple(u)]-W[tuple(v)]))**2).sum()

        d = cdist(I_pts, J_pts, metric=metric)
        assignment = linear_sum_assignment(d)

        # # We can repeat this process with a metric that
        # # makes sure that near I samples are assigned to near
        # # J samples
        # for _ in range(2):
        #     F = (J_pts[assignment[1]] - I_pts)

        #     # perform nearest neighbors on the parameters
        #     _, I_nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree') \
        #         .fit(I_pts) \
        #         .kneighbors(I_pts)
        #     I_nn = I_nn[:, 1]

        #     def metric2(ui, v):
        #         ui = ui[0]
        #         u = I_pts[ui]
        #         F_cost = (((v - u) - F[I_nn[ui]])**2).sum()
        #         return 10*F_cost + (((u - v) * (1+W[tuple(u)]-W[tuple(v)]))**2).sum()

        #     I_temp = np.tile(np.arange(I_pts.shape[0])[:, np.newaxis], (1, 2))
        #     d = cdist(I_temp, J_pts, metric=metric2)
        #     assignment = linear_sum_assignment(d)

        def score(u, v):
            # TODO: make it work in >2D
            I_vals = I[u[:, 1], u[:, 0], np.newaxis]/2
            J_vals = J[v[:, 1], v[:, 0], np.newaxis]/2
            scores = np.sqrt((((u - v)*(I_vals + J_vals))**2).sum(axis=1))
            return np.mean(scores)
        sc = score(I_pts, J_pts[assignment[1]])
        return sc

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, I, J)
            # for each two consecutive frames
            for I, J in zip(video[:-1], video[1:])
        ]
        scores = [f.result() for f in futures]

    return np.mean(scores)


def NCC(video: np.ndarray, ref='previous', return_all=False) -> Union[float, np.ndarray]:
    """Normalized Cross-Correlation score.

    This method works on 2D and 3D video inputs.
    """
    vxm_ncc = vxm.losses.NCC()

    res = []
    for sl in gen_batches(video.shape[0], 128):
        # vxm NCC's assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        frames = tf.convert_to_tensor(video[sl, ..., np.newaxis], dtype=np.float32)
        res += [vxm_ncc.loss(frames[1:], frames[:-1]).numpy().ravel()]
    if return_all:
        return np.concatenate(res)
    else:
        return np.mean(res)


def MSE(video: np.ndarray, ref: str = 'previous', return_all=False) -> Union[float, np.ndarray]:
    """Return MSE score with respect to some reference image.
    
    This method is tested on 2D videos, but should work on 3D videos as well.

    Parameters
    ----------
    video : array
        Contains the video information
    ref : string, optional
        Reference frame/image to use as approx for round-truth.
        Either: previous, median or mean
        Default is 'previous'
    return_all : bool, optional
        whether to return all MSE values for all frames or average the result
        accross frames.

        Defaults to averaging accross frames
    """
    nb_frame_pixels = np.prod(video.shape[1:])
    
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

    if return_all:
        return np.mean((J-I)**2, axis=1)
    else:
        return mean_squared_error(I, J, multioutput='uniform_average')


def COM(video: np.ndarray,
        frame_shape: Optional[tuple] = None,
        threshold: Optional[int] = None,
        return_all=False) -> np.ndarray:
    """Return percentage of frames considered 'failed' due to remoteness to the Center of Mass.

    An important assumption this score makes is that the mean over the centers of mass of
    the input ``video`` is suposed to represent a proper center were all axons are
    visible.
    """
    if frame_shape is None:
        frame_shape = (video.shape[1], video.shape[2])
    if threshold is None:
        threshold = 0.1 * min(frame_shape)

    centers = get_centers(video)

    # consider a frame wrong if the axis is > 10% off the mean
    m_center = centers.mean(axis=0)
    failures = np.sum((centers - m_center)**2, axis=1) > threshold**2
    if return_all:
        return failures
    else:
        return np.sum(failures) / centers.shape[0]


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