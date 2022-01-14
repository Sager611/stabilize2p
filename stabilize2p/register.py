"""Methods used for image registration.

.. moduleauthor:: Adrian Sager <adrian.sagerlaganga@epfl.ch>

"""

import gc
import time
import logging
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np
import tifffile as tiff
import tensorflow as tf
import voxelmorph as vxm
from tqdm import tqdm
from pystackreg import StackReg
from scipy import ndimage as ndi
from sklearn.utils import gen_batches
from tensorflow.keras import backend as K
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path

from . import utils

_LOGGER = logging.getLogger('stabilize2p')


def pysreg_transform(video: np.ndarray, method=StackReg.AFFINE, reference='first') -> np.ndarray:
    """Apply :class:`pystackreg.StackReg` to ``video``."""
    return StackReg(method).register_transform_stack(video, reference=reference)


def com_transform(video: np.ndarray, inplace=False, downsample=2, target=None, return_transform=False, shifts=None) -> np.ndarray:
    """Fast translation alignment of frames based on the Center of Mass.

    This method is not very precise.
    >40 times faster than pystackreg's translation transform.

    Parameters
    ----------
    video : array
        input to align
    inplace : bool, optional
        whether to modify the input ``video`` instead of making a copy.
        Default is False
    downsample : int, optional
        integer factor by which to downscale the frames when calculating the
        CoM and background threshold.

        Default is 2. This improves performance at a very low cost on accuracy
    target : array, optional
        Pixel point target that CoMs must arrive to (must be positive).
        Defaut is None, which means that ``target`` is calculated as the mean
        of the CoMs of the frames of ``video``
    return_transform : bool, optional
        whether to return the calculated shifts.
        Default is False
    shifts : int array, optional
        if provided, do not calculate shifts and simply use this provided ones
        to transform the images.
        Default is None

    Returns
    -------
    array
        copy of the input video (or input ``video`` if ``inplace`` is True), 
        with the translation transform applied
    """
    # if input is a single image, we cannot align it
    if len(video.shape) < 3 or video.shape[0] == 1:
        return video
    if shifts is not None and shifts.dtype != np.int32:
        raise TypeError(f'provided shifts are not int32. They are: {shifts.dtype}')

    if shifts is None:
        # we need to get the center of mass of the _features_,
        # the background should not have any influence
        sparse = video[:, ::downsample, ::downsample]
        cs = utils.get_centers(sparse)
        target = cs.mean(axis=0) if target is None else target / downsample
        shifts = (target - cs) * downsample
        # scipy requires a (row, column) format
        shifts = shifts[:, [1, 0]]
        # VERY IMPORTANT! ndi shift changes the pixels' values if
        # the shifts are floating point! (since it has to interpolate)
        # this value changes can be drastic, so we cast the shifts to int
        shifts = shifts.astype(np.int32)
    # use parallelism
    with ThreadPoolExecutor() as executor:
        returns = None

        if inplace:
            futures = [
                executor.submit(lambda f, s: ndi.shift(f, s, mode='grid-wrap', output=f), frame, shift)
                for frame, shift in zip(video, shifts)
            ]
            for f in futures:
                f.result()
            returns = video
        else:
            futures = [
                executor.submit(lambda f, s: ndi.shift(f, s, mode='grid-wrap'), frame, shift)
                for frame, shift in zip(video, shifts)
            ]
            res = np.array([f.result() for f in futures])
            returns = res

        if return_transform:
            returns = (returns, shifts)
        return returns


def ECC_transform(video: np.ndarray, nb_iters=50, eps=1e-6, ref=None, return_transform=False, warps=None):
    """Affine alignment using :func:`cv2.findTransformECC`.

    >4 times faster than pystackreg's translation transform.

    Parameters
    ----------
    video : array
        input video to align
    ref : array, optional
        reference frame to align to.
        Defaults to first frame in video
    return_transform : bool, optional
        whether to return the warp matrices.
        Default is False
    warps : array, optional
        if provided, warp matrices will not be calculated and instead
        these will be used.
        Default is None
    """
    if ref is None:
        # if there is only one frame, there is nothing to do
        if video.shape[0] <= 1:
            return video.copy()
        ref = video[0]
    ref = ref.astype(np.float32)

    def loop(ref, fi, I):
        I = I.astype(np.float32)
        if warps is None:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, nb_iters, eps)
        else:
            warp_matrix = warps[fi]

        try:
            if warps is None:
                # Run the ECC algorithm. The results are stored in warp_matrix.
                (cc, warp_matrix) = cv2.findTransformECC(ref, I, warp_matrix, cv2.MOTION_AFFINE, criteria)

            # Use warpAffine for Translation, Euclidean and Affine
            I_moved = cv2.warpAffine(I, warp_matrix, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error as e:
            # in case of an OpenCV error, simply return the same frame unmoved
            _LOGGER.error(f'{e}')
            if return_transform:
                return I, warp_matrix
            return I
        
        if return_transform:
            return I_moved, warp_matrix
        return I_moved

    # use parallelism
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(loop, ref, fi, frame)
            for fi, frame in enumerate(video)
        ]
        if return_transform:
            res = [f.result() for f in futures]
            moved = np.array([m for m, _ in res])
            warps = np.array([w for _, w in res])
            return moved, warps
        else:
            res = np.array([f.result() for f in futures])
            return res


def voxelmorph_transform(image_pool: list,
                         voxelmorph_path: str,
                         keys: list = None,
                         return_flow: bool = False,
                         undo_preprocessing_output: bool = True,
                         store_params: list = [],
                         strategy=None,
                         data_generator_kw: dict = {}):
    """Register using a VoxelMorph model.

    Parameters
    ----------
    image_pool : str or list
        path or list of paths to the TIFF files to register
    voxelmorph_path : str
        path to the saved weights of the VoxelMorph model in h5 format
    keys: list, optional
        same argument as for :func:`stabilize2p.utils.vxm_data_generator`, indicates which frames to use for
        each file in the pool. Defaults to use all frames
    return_flow: bool, optional
        whether to also return the predicted flow for each frame.
        Default is False
    undo_preprocessing_output: bool, optional
        whether to undo the pre-processing steps on the VoxelMorph model's output before returning it.
        Default is True
    store_params : list, optional
        if provided, calculated pre-processing parameters for each file in the pool will be stored in this list
    strategy: tensorflow strategy, optional
        which Tensorflow strategy to use to run inference on the model. To get some strategy, you can
        use :func:`stabilize2p.utils.get_strategy`.
        Default is :class:`tf.distribute.MirroredStrategy`
    data_generator_kw : dict, optional
        additional keyword parameters to pass to :func:`stabilize2p.utils.vxm_data_generator`
    Returns
    -------
    array or tuple of arrays
        registered images.

        If ``undo_preprocessing_output`` is False, the outputted images will come from the model's prediction as-is,
        without undoing the applied pre-processing steps.

        If ``return_flow``is True, the predicted flow is also returned.
    """
    if type(image_pool) is str:
        image_pool = [image_pool]

    if strategy is None:
        strategy = utils.get_strategy('GPU')

    # get frame shape
    inshape = tiff.imread(image_pool[0], key=0).shape
    nfeats = 1

    # load model
    K.clear_session()
    with strategy.scope():
        vxm_model = vxm.networks.VxmDense.load(voxelmorph_path)
    _LOGGER.info('model input shape:  ' + ', '.join([str(t.shape) for t in vxm_model.inputs]))
    _LOGGER.info('model output shape: ' + ', '.join([str(t.shape) for t in vxm_model.outputs]))

    # retrieve images as input
    base_gen = utils.vxm_data_generator(image_pool,
                                        keys=keys,
                                        training=False,
                                        store_params=store_params,
                                        **data_generator_kw)
    inputs = [ins for (ins, _) in base_gen]

    del base_gen
    gc.collect()

    moving = np.concatenate([m for m, _ in inputs], axis=0)
    fixed = np.concatenate([f for _, f in inputs], axis=0)

    del inputs
    gc.collect()

    # register
    with strategy.scope():
        if return_flow:
            # Divide in batches otherwise we run out of memory
            moved, flow = [], []
            for idx in tqdm(gen_batches(moving.shape[0], 32)):
                m, f = vxm_model.predict((moving[idx], fixed[idx]))
                moved += [m]
                flow += [f]
            moved = np.concatenate(moved, axis=0).squeeze()
            flow = np.concatenate(flow, axis=0).squeeze()
        else:
            # Divide in batches otherwise we run out of memory
            moved = []
            for idx in tqdm(gen_batches(moving.shape[0], 32)):
                m = vxm_model.predict((moving[idx], fixed[idx]))[0]
                moved += [m]
            moved = np.concatenate(moved, axis=0).squeeze()

    del vxm_model

    if undo_preprocessing_output:
        # undo pre-processing transformations
        i = 0
        for params in store_params:
            idx = slice(i, i+params['nb_frames'])
            moved[idx] = utils.vxm_undo_preprocessing(moved[idx], params)
            i += params['nb_frames']

    if return_flow:
        return moved, flow
    else:
        return moved


def hypermorph_optimal_transform(image_pool: list,
                                 hypermorph_path: str,
                                 keys: list = None,
                                 num_hyp: int = 8,
                                 metric='model-ncc',
                                 return_flow: bool = False,
                                 return_optimal_hyp: bool = False,
                                 undo_preprocessing_output: bool = True,
                                 store_params: list = [],
                                 strategy=None,
                                 data_generator_kw: dict = {}):
    """Register using a HyperMorph model by optimizing its hyperparameter per-frame using some heuristic metric.

    This function calculates :math:`L_{sim}(I_t, I_{t-1})` for each pair of frames :math:`I_t, I_{t-1}` each
    generated with ``num_hyp`` [0, 1] values for the hyperparameter, and with some metric :math:`L_{sim}`.
    Then, the optimal hyperparameters per-frame are calculated by the shortest weighted path from :math:`I_0`
    to :math:`I_{nb\_frames}`. Finally, with these hyperparameters, the images from ``image_pool`` are
    registered and returned as an array.

    Parameters
    ----------
    image_pool : str or list
        path or list of paths to the TIFF files to register
    hypermorph_path : str
        path to the saved weights of the Hypermorph model in h5 format
    keys: list, optional
        same argument as for :func:`stabilize2p.utils.vxm_data_generator`, indicates which frames to use for
        each file in the pool. Defaults to use all frames
    num_hyp: int, optional
        number of hyperparameters between [0, 1] to use. For example, if ``num_hyp=3``, then the considered
        hyperparameters will be ``[0. , 0.5, 1. ]``.
        Default is 8
    metric: str or function, optional
        which similarity metric to use between consecutive frames.
        Default is 'model-ncc', which uses NCC as the similarity loss, and l2 of the flow gradient as an
        additional smoothness loss. In particular, for each image pair :math:`I_t, I_{t-1}` the metric is:
        
        .. math::
        
            0.5 * L_{sim}(I_t, I_{t-1}) + 0.25 * L_{grad}(\phi_t) + 0.25 * L_{grad}(\phi_{t+1})
            
        Where :math:`\phi_t` is the gradient of the predicted flow for :math:`I_t`.
        
        You can instead specify 'model-mse', which will use MSE as the similarity loss.
        
        Further details for other metrics are found in `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_
    return_flow: bool, optional
        whether to also return the predicted flow for each frame.
        Default is False
    return_optimal_hyp: bool, optional
        whether to also return the array of found optimal hyperparameters for each frame.
        Default is False
    undo_preprocessing_output: bool, optional
        whether to undo the pre-processing steps on the Hypermorph model's output before returning it.
        Default is True
    store_params : list, optional
        if provided, calculated pre-processing parameters for each file in the pool will be stored in this list
    strategy: tensorflow strategy, optional
        which Tensorflow strategy to use to run inference on the model. To get some strategy, you can
        use :func:`stabilize2p.utils.get_strategy`.
        Default is :class:`tf.distribute.MirroredStrategy`
    data_generator_kw : dict, optional
        additional keyword parameters to pass to :func:`stabilize2p.utils.vxm_data_generator`
    Returns
    -------
    array or tuple of arrays
        registered images. 

        If ``undo_preprocessing_output`` is False, the outputted images will come from the model's prediction as-is,
        without undoing the applied pre-processing steps.
        
        Depending if ``return_flow`` and/or ``return_optimal_hyp`` are set, the output changes:

        ============================= ============================= =============================
         ~                            ``return_optimal_hyp`` False  ``return_optimal_hyp`` True
        ``return_flow`` False         registered                    (registered, hyps)
        ``return_flow`` True          (registered, flow)            (registered, flow, hyps)
        ============================= ============================= =============================
    """
    if type(image_pool) is str:
        image_pool = [image_pool]

    if strategy is None:
        strategy = utils.get_strategy('GPU')

    # get frame shape
    inshape = tiff.imread(image_pool[0], key=0).shape
    nfeats = 1

    # load model
    K.clear_session()
    with strategy.scope():
        vxm_model = vxm.networks.HyperVxmDense.load(hypermorph_path)
    _LOGGER.info('model input shape:  ' + ', '.join([str(t.shape) for t in vxm_model.inputs]))
    _LOGGER.info('model output shape: ' + ', '.join([str(t.shape) for t in vxm_model.outputs]))

    if metric.startswith('model'):
        # prepare loss functions
        image_loss = metric.split('-')[1].lower()
        if image_loss == 'ncc':
            ncc_loss = vxm.losses.NCC().loss
            def image_loss(I, J):
                with ThreadPoolExecutor() as executor:
                    I = [tf.convert_to_tensor(np.repeat([x], num_hyp, axis=0)) for x in I]
                    J = tf.convert_to_tensor(J)
                    futures = [
                        executor.submit(ncc_loss, x, J)
                        for x in I
                    ]
                    scores = np.array([f.result() for f in futures])
                    return scores
        elif image_loss == 'mse':
            # loss parameters
            image_sigma = 0.05  # a priori image noise std

            scaling = 1.0 / (image_sigma ** 2) / np.prod(inshape)
            image_loss = lambda I, J: scaling * cdist(I.reshape((num_hyp, -1)), J.reshape((num_hyp, -1)), metric='sqeuclidean')

        int_downsize = vxm_model.outputs[0].shape[1] // vxm_model.outputs[1].shape[1]
        grad_loss = vxm.losses.Grad('l2', loss_mult=int_downsize).loss

    # retrieve images as input
    base_gen = utils.vxm_data_generator(image_pool,
                                        keys=keys,
                                        training=False,
                                        store_params=store_params,
                                        **data_generator_kw)
    inputs = [ins for (ins, _) in base_gen]

    del base_gen
    gc.collect()

    moving = np.concatenate([m for m, _ in inputs], axis=0)
    fixed = np.concatenate([f for _, f in inputs], axis=0)

    del inputs
    gc.collect()

    # compute metric scores for all consecutive frame pairs
    # as a sparse matrix representing a graph, so we can find the
    # shortest paths later
    t1 = time.perf_counter()
    # shape: (num_hyp, 1)
    hyper_params = np.linspace(0, 1, num_hyp)[:, np.newaxis]
    sg_data = []
    sg_indices = []
    sg_indptr = []
    in_hyp = hyper_params[:, np.newaxis]
    in_moving = np.repeat([moving[0]], num_hyp, axis=0)
    in_fixed = np.repeat([fixed[0]], num_hyp, axis=0)
    with strategy.scope():
        if metric.startswith('model'):
            # shape: (num_hyp, W, H, 1)
            I, I_flow = vxm_model.predict((in_moving, in_fixed, in_hyp))
            I = I.squeeze()
            I_flow = tf.convert_to_tensor(I_flow)
            I_flow_loss = np.squeeze(grad_loss(None, I_flow))
        else:
            # shape: (num_hyp, W, H)
            I = vxm_model.predict((in_moving, in_fixed, in_hyp))[0].squeeze()
            I = I.reshape((num_hyp, -1))
        for t in tqdm(range(moving.shape[0]-1)):
            # model forward pass
            in_moving = np.repeat([moving[t+1]], num_hyp, axis=0)
            in_fixed = np.repeat([fixed[t+1]], num_hyp, axis=0)
            if metric.startswith('model'):
                # shape: (num_hyp, W, H, 1)
                # TODO: change to predict_on_batch may improve performance at the risk of OOM
                J, J_flow = vxm_model.predict((in_moving, in_fixed, in_hyp))
                J = J.squeeze()
                J_flow = tf.convert_to_tensor(J_flow)
                J_flow_loss = np.squeeze(grad_loss(None, J_flow))
            else:
                # shape: (num_hyp, W, H)
                J = vxm_model.predict((in_moving, in_fixed, in_hyp))[0].squeeze()
                J = J.reshape((num_hyp, -1))

            if metric.startswith('model'):
                # 0.5 * MSE(I, J) + 0.25 * flow_loss(I) + 0.25 * flow_loss(J)
                #  or
                # 0.5 * NCC(I, J) + 0.25 * flow_loss(I) + 0.25 * flow_loss(J)
                scores = 0.5 * image_loss(I, J) + 0.25 * I_flow_loss[:, np.newaxis] + 0.25 * J_flow_loss[np.newaxis, :]
            else:
                scores = cdist(I, J, metric=metric)
            # scores has as rows input nodes and as cols output nodes
            sg_indices = np.r_[sg_indices,
                               np.tile((t+1)*num_hyp + np.arange(num_hyp), num_hyp)]
            sg_data = np.r_[sg_data,
                            scores.flatten()]

            I = J
            if metric.startswith('model'):
                I_flow_loss = J_flow_loss
    # delete unnecessary memory
    del I, J
    if metric.startswith('model'):
        del I_flow_loss, J_flow_loss

    sg_indptr = np.arange(0, sg_indices.size+1, num_hyp)
    # last image's hyp nodes are sinks, so we add their rows as empty
    sg_indptr = np.r_[sg_indptr, np.repeat(sg_indptr[-1], num_hyp)]

    # cast as int type
    sg_indptr = sg_indptr.astype('int32')
    sg_indices = sg_indices.astype('int32')

    # a Compressed Sparse Row matrix
    N = moving.shape[0]*num_hyp
    score_graph = csr_matrix((sg_data, sg_indices, sg_indptr), shape=(N, N))
    t2 = time.perf_counter()
    _LOGGER.info(f'Calcuated scores. Elapsed {t2-t1:.2f}s '
                 f'| {moving.shape[0]/(t2-t1):.0f} frames/s '
                 f'| {(t2-t1)/moving.shape[0]:.4f} s/frame')

    t1 = time.perf_counter()
    # dist_matrix shape: (num_hyp, N_nodes)
    dist_matrix, predecessors = \
        shortest_path(score_graph,
                      directed=True,
                      return_predecessors=True,
                      indices=np.arange(num_hyp))
    t2 = time.perf_counter()
    _LOGGER.info(f'Calcuated shortest paths. Elapsed {t2-t1:.4f}s')
    _LOGGER.info(f'Shortest path scores: \n{dist_matrix[:, -num_hyp:]}')

    # build path of hyp using predecessors
    optimal_hyp = []
    opt_idx_source_sink = np.argmin(dist_matrix[:, -num_hyp:])
    opt_idx_source = opt_idx_source_sink // num_hyp
    # we are only interested in the source with optimal path.
    # ``predecessors`` will now be just its row
    predecessors = predecessors[opt_idx_source]
    
    optimal_hyp.append(hyper_params[opt_idx_source_sink % num_hyp])
    opt_idx = predecessors[-num_hyp:][opt_idx_source_sink % num_hyp]
    while opt_idx != -9999:
        optimal_hyp.append(hyper_params[opt_idx % num_hyp])
        opt_idx = predecessors[opt_idx]
    # we have been adding optimal hyps from the last frame to the first,
    # so we need to reverse the list
    optimal_hyp = np.array(optimal_hyp)[::-1]
    optimal_hyp = optimal_hyp.squeeze()
    _LOGGER.info(f'optimal hyps: {optimal_hyp}')
    
    del score_graph, sg_data, sg_indices, sg_indptr, dist_matrix, predecessors

    # finally! register result
    # shape: (N_frames, W, H) | flow shape: (N_frames, W, H, N_dimensions)
    optimal_hyp = optimal_hyp[:, np.newaxis]
    with strategy.scope():
        if return_flow:
            # Divide in batches otherwise we get a weird CPU to GPU internal error: Dst tensor is not initialized.
            moved, flow = [], []
            for idx in tqdm(gen_batches(moving.shape[0], 32)):
                m, f = vxm_model.predict((moving[idx], fixed[idx], optimal_hyp[idx]))
                moved += [m]
                flow += [f]
            moved = np.concatenate(moved, axis=0).squeeze()
            flow = np.concatenate(flow, axis=0).squeeze()
        else:
            # Divide in batches otherwise we get a weird CPU to GPU internal error: Dst tensor is not initialized.
            moved = []
            for idx in tqdm(gen_batches(moving.shape[0], 32)):
                m = vxm_model.predict((moving[idx], fixed[idx], optimal_hyp[idx]))[0]
                moved += [m]
            moved = np.concatenate(moved, axis=0).squeeze()
    del vxm_model

    if undo_preprocessing_output:
        # undo pre-processing transformations
        i = 0
        for params in store_params:
            idx = slice(i, i+params['nb_frames'])
            moved[idx] = utils.vxm_undo_preprocessing(moved[idx], params)
            i += params['nb_frames']

    returns = moved
    if return_flow:
        returns = (returns, flow)
    if return_optimal_hyp:
        if type(returns) is tuple:
            returns = (*returns, optimal_hyp)
        else:
            returns = (returns, optimal_hyp)
    
    return returns
