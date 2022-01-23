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


def EMD(video: np.ndarray,
        ref='previous',
        n_samples: int = 500,
        metric: str = 'euclidean',
        return_all=False) -> Union[float, np.ndarray]:
    """Earth Moving Distance score.
    
    This metric is not standard and it is probabilistic.

    This is a generalization for the 1D Wasserstein distance.

    ``n_samples`` samples are taken for each frame, considering the frame
    an n-dimensional distribution. Then, for each two consecutive frames,
    their sampled points distances are run through :func:`scipy.optimize.linear_sum_assignment`.
    
    .. note::

        Check the following paper for theoretical and experimental results on this approach:

        Bharath K Sriperumbudur et al. “On the empirical estimation of integral probabilitymetrics”. In: *Electronic Journal of Statistics* 6 (2012), pp. 1550–1599.

    Parameters
    ----------
    video : array
        n-dimensional video
    ref : string or array, optional
        Reference frame/image to use as approx for round-truth.
        Either: 'previous', 'first', or an array
        Default is 'previous'
    n_samples : array
        number of samples to take for each frame.
        Defaults to 100.
    metric : string
        distance metric to use for optimal transport.
        Defaults to 'euclidean'
    return_all : bool, optional
        whether to return all EMD values for all frames or average the result
        across frames.

        Defaults to averaging across frames

    Returns
    -------
    float or array
        average NCC score across frames, or all NCC scores per-frame if ``return_all`` is True
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

        # sample random points
        I_idx = np.random.choice(I_flat.size, p=I_flat, size=n_samples)
        J_idx = np.random.choice(J_flat.size, p=J_flat, size=n_samples)

        I_pts = np.c_[np.unravel_index(I_idx, I.shape)]
        J_pts = np.c_[np.unravel_index(J_idx, J.shape)]

        # calculate minimum distance assignment
        d = cdist(I_pts, J_pts, metric=metric)
        assignment = linear_sum_assignment(d)

        return d[assignment].sum() / n_samples

    with ThreadPoolExecutor() as executor:
        if type(ref) is np.ndarray:
            futures = [
                executor.submit(loop, I, J)
                for I, J in zip(video, ref)
            ]
        elif ref == 'previous':
            futures = [
                executor.submit(loop, I, J)
                # for each two consecutive frames
                for I, J in zip(video[:-1], video[1:])
            ]
        elif ref == 'first':
            futures = [
                executor.submit(loop, I, video[0])
                for I in video
            ]
        else:
            raise ValueError(f'Reference "{ref}" is not recognized. Recognized references: previous, first')
        scores = [f.result() for f in futures]

    if return_all:
        return scores
    else:
        return np.mean(scores)


def NCC(video: np.ndarray, ref='previous', return_all=False) -> Union[float, np.ndarray]:
    """Normalized Cross-Correlation score.

    This method works on 2D and 3D video inputs.

    Parameters
    ----------
    video : array
        Contains the video information
    ref : string or array, optional
        Reference frame/image to use as approx for round-truth.
        Either: 'previous', 'first' or an array
        Default is 'previous'
    return_all : bool, optional
        whether to return all NCC values for all frames or average the result
        across frames.

        Defaults to averaging across frames

    Returns
    -------
    float or array
        average NCC score across frames, or all NCC scores per-frame if ``return_all`` is True
    """
    vxm_ncc = vxm.losses.NCC()

    res = []
    for sl in gen_batches(video.shape[0], 128):
        # vxm NCC's assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        frames = tf.convert_to_tensor(video[sl, ..., np.newaxis], dtype=np.float32)
        if type(ref) is np.ndarray:
            ref_frames = tf.convert_to_tensor(ref[sl, ..., np.newaxis], dtype=np.float32)
            res += [vxm_ncc.loss(frames, ref_frames).numpy().squeeze()]
        elif ref == 'previous':
            res += [vxm_ncc.loss(frames[1:], frames[:-1]).numpy().squeeze()]
        elif ref == 'first':
            ref_frames = tf.tile(
                frames[0][np.newaxis, ...],
                [frames.shape[0]] + [1 for _ in range(len(frames.shape)-1)]
            )
            res += [vxm_ncc.loss(frames, ref_frames).numpy().squeeze()]
            del ref_frames
        else:
            raise ValueError(f'Reference "{ref}" is not recognized. Recognized references: previous, first')
        del frames

    if len(res) > 1:
        res = np.concatenate(res)

    if return_all:
        return res
    else:
        # print(f'{res.shape=}')
        return np.mean(res)


def MSE(video: np.ndarray, ref='previous', return_all=False) -> Union[float, np.ndarray]:
    """Return MSE score with respect to some reference image.
    
    This method is tested on 2D videos, but should work on 3D videos as well.

    Parameters
    ----------
    video : array
        Contains the video information
    ref : string or array, optional
        Reference frame/image to use as approx for round-truth.
        Either: 'previous', 'first', 'median', 'mean' or an array
        Default is 'previous'
    return_all : bool, optional
        whether to return all MSE values for all frames or average the result
        across frames.

        Defaults to averaging across frames

    Returns
    -------
    float or array
        average MSE score across frames, or all MSE scores per-frame if ``return_all`` is True
    """
    nb_frame_pixels = np.prod(video.shape[1:])
    
    if type(ref) is np.ndarray:
        I = ref.reshape((-1, nb_frame_pixels))
    elif ref == 'previous':
        I = video[:-1].reshape((-1, nb_frame_pixels))
    elif ref == 'first':
        I = np.tile(video[0].ravel(), (video.shape[0], 1))
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
        return_all=False) -> Union[float, np.ndarray]:
    """Return fraction of frames considered 'failed' due to remoteness to the Center of Mass.

    An important assumption this score makes is that the mean over the centers of mass of
    the input ``video`` is suposed to represent a proper center were all axons are
    visible.

    Parameters
    ----------
    video : array
        contains the video information
    frame_shape : tuple, optional
        (width, height) of frames in pixels. Used to calculate the ``threshold``, if it is not
        provided, using the formula:

        .. math::
            
            threshold = 10\% \cdot \min\{width, height\}

        Default is the width and height of the frames in ``video``
    threshold : float, optional
        radius used to consider a frame as *failed*.
        Default is :math:`threshold = 10\% \cdot \min\{width, height\}`, where (width, height) are defined by
        ``frame_shape``
    return_all : bool, optional
        whether to return all MSE values for all frames or average the result
        across frames.

        Defaults to averaging across frames
    
    Returns
    -------
    float or array
        average COM score across frames, or all COM scores per-frame if ``return_all`` is True
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
