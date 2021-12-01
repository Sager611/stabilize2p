"""Methods and classes used for image registration.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

import time
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
# from skimage.measure import block_reduce

from . import utils


def com_transform(video: np.ndarray, inplace=False, downsample=2) -> np.ndarray:
    """Fast translation alignment of frames based on the Center of Mass.

    >30 times faster than pystackreg's translation transform.

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

    Returns
    -------
    np.ndarray
        copy of the input video (or input ``video`` if ``inplace`` is True), 
        with the translation transform applied
    """
    # we need to get the center of mass of the _features_,
    # the background should not have any influence
    sparse = video[:, ::downsample, ::downsample]
    th = utils.estimate_background_threshold(sparse)
    cs = utils.get_centers(sparse - th)
    mean_center = cs.mean(axis=0)
    shifts = (mean_center - cs) * downsample
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
    