"""Methods used for image registration.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

import time
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np
from pystackreg import StackReg
from scipy import ndimage as ndi

from . import utils


def pysreg_transform(video: np.ndarray, method=StackReg.AFFINE, reference='first') -> np.ndarray:
    """Apply :class:`pystackreg.StackReg` to ``video``."""
    return StackReg(method).register_transform_stack(video, reference=reference)


def com_transform(video: np.ndarray, inplace=False, downsample=2, target=None) -> np.ndarray:
    """Fast translation alignment of frames based on the Center of Mass.

    This method is not very precise.
    >40 times faster than pystackreg's translation transform.

    Parameters
    ----------
    video : array
        input to align
    inplace : bool
        whether to modify the input ``video`` instead of making a copy.
        Default is False
    downsample : int
        integer factor by which to downscale the frames when calculating the
        CoM and background threshold.

        Default is 2. This improves performance at a very low cost on accuracy
    target : array
        Pixel point target that CoMs must arrive to (must be positive).
        Defaut is None, which means that ``target`` is calculated as the mean
        of the CoMs of the frames of ``video``

    Returns
    -------
    array
        copy of the input video (or input ``video`` if ``inplace`` is True), 
        with the translation transform applied
    """
    # if input is a single image, we cannot align it
    if len(video.shape) < 3 or video.shape[0] == 1:
        return video

    # we need to get the center of mass of the _features_,
    # the background should not have any influence
    sparse = video[:, ::downsample, ::downsample]
    th = utils.estimate_background_threshold(video[0])
    cs = utils.get_centers(sparse - th)
    target = cs.mean(axis=0) if target is None else target / 2
    shifts = (target - cs) * downsample
    # scipy requires a (row, column) format
    shifts = shifts[:, [1, 0]]
    # use parallelism
    with ThreadPoolExecutor() as executor:
        if inplace:
            futures = [
                executor.submit(lambda f, s: ndi.shift(f, s, mode='grid-wrap', output=f), frame, shift)
                for frame, shift in zip(video, shifts)
            ]
            for f in futures:
                f.result()
            return video
        else:
            futures = [
                executor.submit(lambda f, s: ndi.shift(f, s, mode='grid-wrap'), frame, shift)
                for frame, shift in zip(video, shifts)
            ]
            res = np.array([f.result() for f in futures])
            return res

def ECC_transform(video: np.ndarray, nb_iters=5, eps=1e-6, ref=None) -> np.ndarray:
    """Affine alignment using :func:`cv2.findTransformECC`.

    >4 times faster than pystackreg's translation transform.

    Parameters
    ----------
    video : array
        input video to align
    ref : array, optional
        reference frame to align to.
        Defaults to first frame in video
    """
    if ref is None:
        ref = video[0]
    ref = ref.astype(np.float32)

    def loop(ref, I):
        I = I.astype(np.float32)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, nb_iters, eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(ref, I, warp_matrix, cv2.MOTION_AFFINE, criteria)

        # Use warpAffine for Translation, Euclidean and Affine
        I_moved = cv2.warpAffine(I, warp_matrix, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return I_moved

    # use parallelism
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, ref, frame)
            for frame in video
        ]
        res = np.array([f.result() for f in futures])
        return res