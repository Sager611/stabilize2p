"""Module including thresholding methods.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

import logging

import cv2
import numpy as np

_LOGGER = logging.getLogger('stabilize2p')


def otsu(image: np.ndarray, bins=255) -> float:
    """Calculate Otsu threshold.
    
    .. note::
    
        Otsu, Nobuyuki. “A threshold selection method from gray level histograms.” *IEEE Transactions on Systems, Man, and Cybernetics* 9 (1979): 62-66.
    """
    otsu, _ = cv2.threshold(image.astype(np.float32), 0, bins, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def watershed(image: np.ndarray, num_peaks: int = 2, bins: int = 400) -> float:
    """Apply the watershed algorithm to find threshold between the first two histogram modes.

    In this method, it is assumed that the pixel values of ``image`` form a multi-modal
    histogram, in which the lower mode corresponds to background pixels and the 2nd lowest one
    is the foreground. Thus, the threshold is estimated to be in the valley between
    these two.

    Parameters
    ----------
    image : array
        2D image
    num_peaks : int, optional
        a-priori number of peaks that the pixel value histogram of ``image`` has
    bins : int, optional
        number of bins to use for the pixel value histogram of ``image``
    """
    pix_hist, bns = np.histogram(image.ravel(), bins=bins)
    # we assume by default num_peaks=2, that is, we have a bimodal pixel value histogram
    coords = peak_local_max(pix_hist, num_peaks=num_peaks)
    hig = pix_hist.argmax()

    # if the distribution is uni-modal (in which case we assume that the 2nd predicted peak is small)
    if pix_hist[coords[1]]/pix_hist[coords[0]] < 0.1:
        threshold_i = int(coords[0])
        _LOGGER.warning('pixel histogram is uni-modal, estimated threshold may not be accurate.')
    else:
        mask = np.zeros(pix_hist.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)

        ws = watershed(-pix_hist, markers)
        idx = np.where(ws[1:] > ws[:-1])[0]

        coords = np.sort(coords.ravel())
        if idx.size == 0:
            threshold_i = int(coords[0])
        else:
            # threshold is between the two first local maxima
            assert np.sum((idx >= coords[0]) & (idx <= coords[1])) > 0, f'{coords=} | {idx=}'
            threshold_i = idx[(idx >= coords[0]) & (idx <= coords[1])].min()
    return bns[threshold_i:(threshold_i+1)].mean()