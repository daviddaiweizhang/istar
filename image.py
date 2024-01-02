import numpy as np
import cv2 as cv
import torch
from torch import nn
import skimage
from scipy.ndimage import uniform_filter


def impute_missing(x, mask, radius=3, method='ns'):

    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]

    x = x.copy()
    if x.dtype == np.float64:
        x = x.astype(np.float32)

    x[mask] = 0
    mask = mask.astype(np.uint8)

    expand_dim = np.ndim(x) == 2
    if expand_dim:
        x = x[..., np.newaxis]
    channels = [x[..., i] for i in range(x.shape[-1])]
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    if expand_dim:
        y = y[..., 0]

    return y


def smoothen(
        x, size, kernel='gaussian', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'gaussian':
        sigma = size / 4  # approximate std of uniform filter 1/sqrt(12)
        truncate = 4.0
        winsize = np.ceil(sigma * truncate).astype(int) * 2 + 1
        if backend == 'cv':
            print(f'gaussian filter: winsize={winsize}, sigma={sigma}')
            y = cv.GaussianBlur(
                    x, (winsize, winsize), sigmaX=sigma, sigmaY=sigma,
                    borderType=cv.BORDER_REFLECT)
        elif backend == 'skimage':
            y = skimage.filters.gaussian(
                    x, sigma=sigma, truncate=truncate,
                    preserve_range=True, channel_axis=-1)
        else:
            raise ValueError('backend must be cv or skimage')
    elif kernel == 'uniform':
        if backend == 'cv':
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        elif backend == 'torch':
            assert isinstance(size, int)
            padding = size // 2
            size = size + 1

            pool_dict = {
                    'mean': nn.AvgPool2d(
                        kernel_size=size, stride=1, padding=0),
                    'max': nn.MaxPool2d(
                        kernel_size=size, stride=1, padding=0)}
            pool = pool_dict[mode]

            mod = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    pool)
            y = mod(torch.tensor(x, device=device).permute(2, 0, 1))
            y = y.permute(1, 2, 0)
            y = y.cpu().detach().numpy()
        else:
            raise ValueError('backend must be cv or torch')
    else:
        raise ValueError('kernel must be gaussian or uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]

    return y


def upscale(x, target_shape):
    mask = np.isfinite(x).all(tuple(range(2, x.ndim)))
    x = impute_missing(x, ~mask, radius=3)
    # TODO: Consider using pytorch with cuda to speed up
    # order: 0 == nearest neighbor, 1 == bilinear, 3 == bicubic
    dtype = x.dtype
    x = skimage.transform.resize(
            x, target_shape, order=3, preserve_range=True)
    x = x.astype(dtype)
    if not mask.all():
        mask = skimage.transform.resize(
                mask.astype(float), target_shape, order=3,
                preserve_range=True)
        mask = mask > 0.5
        x[~mask] = np.nan
    return x


def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img


def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin


def shrink_mask(x, size):
    size = size * 2 + 1
    x = uniform_filter(x.astype(float), size=size)
    x = np.isclose(x, 1)
    return x
