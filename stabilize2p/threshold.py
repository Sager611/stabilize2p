"""Module including thresholding methods.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

import logging

import cv2
import skimage.segmentation
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

_LOGGER = logging.getLogger('stabilize2p')


def otsu(image: np.ndarray) -> float:
    """Calculate Otsu threshold.
    
    .. seealso::
    
        Otsu, Nobuyuki. “A threshold selection method from gray level histograms.” *IEEE Transactions on Systems, Man, and Cybernetics* 9 (1979): 62-66.
    """
    # normalize image to use in cv2.threshold
    low, hig = image.min(), image.max()
    image = (image - low) / (hig - low) * 255
    image = image.astype(np.uint8)

    # Otsu improves with a Gaussian blur
    k = int(0.01 * max(image.shape))
    blur = cv2.GaussianBlur(image, (k, k), 0)

    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    return (otsu / 255) * (hig - low) + low


def triangle(image: np.ndarray) -> float:
    """Calculate threshold using OpenCV's triangle method.

    If ``image`` is a 2D video, this function calculates the threshold at ~1600 frames/s.

    .. seealso::

        Zack, G W et al. “Automatic measurement of sister chromatid exchange frequency.” *The journal of histochemistry and cytochemistry : official journal of the Histochemistry Society* vol. 25,7 (1977): 741-53. doi:10.1177/25.7.70454

    Parameters
    ----------
    image : array
        can be a video
    """
    # normalize image to use in cv2.threshold
    low, hig = image.min(), image.max()
    image = (image - low) / (hig - low) * 255
    image = image.astype(np.uint8)

    thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_TRIANGLE)
    return (thresh / 255) * (hig - low) + low


def watershed(image: np.ndarray, num_peaks: int = 2, bins: int = 800) -> float:
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

    # if the distribution is uni-modal (in which case we assume that the 2nd predicted peak is small)
    if pix_hist[coords[1]]/pix_hist[coords[0]] < 0.1:
        threshold_i = int(coords[0])
        _LOGGER.warning('pixel histogram is uni-modal, estimated threshold may not be accurate.')
    else:
        mask = np.zeros(pix_hist.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)

        ws = skimage.segmentation.watershed(-pix_hist, markers)
        idx = np.where(ws[1:] > ws[:-1])[0]

        coords = np.sort(coords.ravel())
        if idx.size == 0:
            threshold_i = int(coords[0])
        else:
            # threshold is between the two first local maxima
            assert np.sum((idx >= coords[0]) & (idx <= coords[1])) > 0, f'{coords=} | {idx=}'
            threshold_i = idx[(idx >= coords[0]) & (idx <= coords[1])].min()
    return bns[threshold_i:(threshold_i+1)].mean()


def deriche(image: np.ndarray) -> float:
    """Return second maximum of histogram's second derivative using Deriche filter.

    .. seealso::

        Collewet, G et al. “Influence of MRI acquisition protocols and image intensity normalization methods on texture classification.” *Magnetic resonance imaging* vol. 22,1 (2004): 81-91. doi:10.1016/j.mri.2003.09.001
    """
    raise NotImplementedError()


def second_deriv(image: np.ndarray, bins: int = 800) -> float:
    """Return 2nd maximum of histogram's second derivative.

    Parameters
    ----------
    image : array
        2D image
    bins : int, optional
        number of bins to use for the pixel value histogram of ``image``
    """
    hist, bns = np.histogram(image, bins=400)

    dx = bns[1] - bns[0]
    grad = (hist[1:] - hist[:-1]) / dx
    grad = cv2.GaussianBlur(grad, (1, 31), 0)

    grad2 = (grad[1:] - grad[:-1]) / dx
    grad2 = cv2.GaussianBlur(grad2, (1, 31), 0)

    coords = peak_local_max(grad2.ravel(), num_peaks=2)

    i = coords.max()
    return bns[i:(i+1)].mean()