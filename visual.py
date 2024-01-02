import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from einops import rearrange

from utils import save_image
from image import get_disk_mask


def cmap_myset(x):
    cmap = ListedColormap([
        '#17BECF',  # cyan
        '#FD8D3C',  # orange
        '#A6D854',  # light green
        '#9467BD',  # purple
        '#A5A5A5',  # gray
        '#F4CAE4',  # light pink
        '#C47A3D',  # brown
        '#FFF800',  # yellow
        ])
    return cmap(x)


def cmap_accent(x):
    cmap = ListedColormap([
        '#386cb0',  # blue
        '#fdc086',  # orange
        '#7fc97f',  # green
        '#beaed4',  # purple
        '#f0027f',  # magenta
        '#bf5b17',  # brown
        '#666666',  # gray
        '#ffff99',  # yellow
        ])
    return cmap(x)


def cmap_turbo_adj(x):

    a = 0.70  # lightness adjustment
    b = 0.70  # satuation adjustment

    cmap = plt.get_cmap('turbo')
    x = np.array(cmap(x))[..., :3]
    x = 1 - (1 - x) * a
    lightness = x.mean(-1, keepdims=True)
    x = (x - lightness) * b + lightness
    return x


def cmap_turbo_truncated(x):
    cmap = plt.get_cmap('turbo')
    x = x * 0.9 + 0.05
    y = np.array(cmap(x))[..., :3]
    return y


def cmap_tab30(x):
    n_base = 20
    n_max = 30
    brightness = 0.7
    brightness = (brightness,) * 3 + (1.0,)
    isin_base = (x < n_base)[..., np.newaxis]
    isin_extended = ((x >= n_base) * (x < n_max))[..., np.newaxis]
    isin_beyond = (x >= n_max)[..., np.newaxis]
    color = (
        isin_base * cmap_tab20(x)
        + isin_extended * cmap_tab20(x-n_base) * brightness
        + isin_beyond * (0.0, 0.0, 0.0, 1.0))
    return color


def cmap_tab70(x):
    cmap_base = cmap_tab30
    brightness = 0.5
    brightness = np.array([brightness] * 3 + [1.0])
    color = [
        cmap_base(x),  # same as base colormap
        1 - (1 - cmap_base(x-20)) * brightness,  # brighter
        cmap_base(x-20) * brightness,  # darker
        1 - (1 - cmap_base(x-40)) * brightness**2,  # even brighter
        cmap_base(x-40) * brightness**2,  # even darker
        [0.0, 0.0, 0.0, 1.0],  # black
        ]
    x = x[..., np.newaxis]
    isin = [
        (x < 30),
        (x >= 30) * (x < 40),
        (x >= 40) * (x < 50),
        (x >= 50) * (x < 60),
        (x >= 60) * (x < 70),
        (x >= 70)]
    color_out = np.sum(
            [isi * col for isi, col in zip(isin, color)],
            axis=0)
    return color_out


def interlaced_cmap(cmap, stop, stride, start=0):

    def cmap_new(x):
        isin = x >= 0
        x[isin] = (x[isin] * stride + start) % stop
        return cmap(x)

    return cmap_new


def reversed_cmap(cmap, stop):

    def cmap_new(x):
        isin = x >= 0
        x[isin] = stop - x[isin] - 1
        return cmap(x)

    return cmap_new


def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)


def get_cmap_tab_multi(n_colors, n_shades, paired=True):
    cmap_base = cmap_tab20
    n_base = 10
    assert n_colors <= n_base

    def cmap(x):
        isin = x >= 0
        x = x * isin

        # lightness
        i = x // n_colors
        if paired:
            is_odd = i % 2 == 1
            i[~is_odd] //= 2
            i[is_odd] = n_shades - 1 - (i[is_odd] - 1) // 2

        # hue
        j = x % n_colors
        colors = np.stack(
                [cmap_base(k)[..., :3] for k in [j, j+n_base]])

        # compute color from hue and lightness
        weights = 1 - i / max(1, n_shades - 1)
        weights = np.stack([weights, 1-weights])
        col = (weights[..., np.newaxis] * colors).sum(0)
        col[~isin] = 0
        return col

    return cmap


def get_cmap_discrete(n_colors, cmap_name):
    cmap_base = plt.get_cmap(cmap_name)

    def cmap(x):
        x = x / float(n_colors-1)
        return cmap_base(x)

    return cmap


def plot_labels(
        labels, filename, cmap=None, white_background=True,
        transparent_background=False,
        interlace=False, reverse=False):
    if labels.ndim == 3:
        n_labels = labels[..., 0].max() + 1
        n_shades = labels[..., 1].max() + 1
        isin = (labels >= 0).all(-1)
        labels_uni = labels[..., 0].copy()
        labels_uni[isin] = (
                n_labels * labels[isin][..., -1]
                + labels[isin][..., 0])
        labels = labels_uni
    elif labels.ndim == 2:
        n_labels = labels.max() + 1
        n_shades = 1

    if cmap is None:
        if n_labels <= 70:
            cmap = 'tab70'
        else:
            cmap = 'turbo'

    if cmap == 'tab70':
        cmap = cmap_tab70
    elif cmap == 'turbo':
        cmap = plt.get_cmap('turbo')
        labels = labels / labels.max()
    elif cmap == 'multi':
        cmap = get_cmap_tab_multi(n_labels, n_shades)
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if interlace:
        cmap = interlaced_cmap(
                cmap, stop=n_labels, start=0, stride=9)
    if reverse:
        cmap = reversed_cmap(cmap, stop=n_labels)

    image = cmap(labels)

    mask_extra = labels < 0
    mask_background = (labels == labels.min()) * mask_extra
    image[mask_extra] = 0.5  # gray
    if white_background:
        background_color = 1.0  # white
    else:
        background_color = 0.0  # black
    image[mask_background] = background_color
    if transparent_background:
        image[mask_background, -1] = 0.0
    image = (image * 255).astype(np.uint8)

    if filename is not None:
        save_image(image, filename)

    return image


def plot_embeddings(embeddings, prefix, groups=None, same_color_scale=True):
    if groups is None:
        groups = embeddings.keys()
    cmap = plt.get_cmap('turbo')
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for key in groups:
        emb = embeddings[key]
        mask = np.all([np.isfinite(channel) for channel in emb], 0)
        min_all = np.min([channel[mask].min() for channel in emb], 0)
        max_all = np.max([channel[mask].max() for channel in emb], 0)
        for i, channel in enumerate(emb):
            if same_color_scale:
                min_chan, max_chan = min_all, max_all
            else:
                min_chan = channel[mask].min()
                max_chan = channel[mask].max()
            image = (channel - min_chan) / (max_chan - min_chan)
            image = cmap(image)[..., :3]
            if not mask.all():
                image[~mask] = 0.0
            image = Image.fromarray((image * 255).astype(np.uint8))
            outfile = f'{prefix}{key}-{i:02d}.png'
            image.save(outfile)
            print(outfile)


def plot_spots(
        img, cnts, locs, radius, outfile, cmap='magma',
        weight=0.8, disk_mask=True, standardize_img=False):
    cnts = cnts.astype(np.float32)

    img = img.astype(np.float32)
    img /= 255.0

    if standardize_img:
        if np.isclose(0.0, np.nanstd(img, (0, 1))).all():
            img[:] = 1.0
        else:
            img -= np.nanmin(img)
            img /= np.nanmax(img) + 1e-12

    cnts -= np.nanmin(cnts)
    cnts /= np.nanmax(cnts) + 1e-12

    cmap = plt.get_cmap(cmap)
    if disk_mask:
        mask_patch = get_disk_mask(radius)
    else:
        mask_patch = np.ones((radius*2, radius*2)).astype(bool)
    indices_patch = np.stack(np.where(mask_patch), -1)
    indices_patch -= radius
    for ij, ct in zip(locs, cnts):
        color = np.array(cmap(ct)[:3])
        indices = indices_patch + ij
        img[indices[:, 0], indices[:, 1]] *= 1 - weight
        img[indices[:, 0], indices[:, 1]] += color * weight
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)


def plot_label_masks(labels, prefix, names=None, white_background=True):

    cmap = plt.get_cmap('tab10')
    color_pos = cmap(1)[:3]  # orange
    color_neg = (0.8, 0.8, 0.8)  # gray
    color_bg = 0.0  # black
    if white_background:
        color_bg = 1.0  # white

    foreground = labels >= 0
    labs_uniq = np.unique(labels)
    labs_uniq = labs_uniq[labs_uniq >= 0]
    if names is None:
        names = [f'{i:03d}' for i in labs_uniq]

    for lab in labs_uniq:
        mask = labels == lab
        img = np.zeros(mask.shape+(3,), dtype=np.float32)
        img[mask] = color_pos
        img[~mask] = color_neg
        img[~foreground] = color_bg
        img = (img * 255).astype(np.uint8)
        nam = names[lab]
        save_image(img, f'{prefix}{nam}.png')


def mat_to_img(
        x, white_background=True, transparent_background=False,
        cmap='turbo', minmax=None):
    mask = np.isfinite(x)
    x = x.astype(np.float32)
    if minmax is None:
        minmax = (np.nanmin(x), np.nanmax(x) + 1e-12)
    print('minmax:', minmax)
    x -= minmax[0]
    x /= minmax[1] - minmax[0]
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    x = cmap(x)
    if white_background:
        x[~mask] = 1.0
    if transparent_background:
        x[~mask, -1] = 0.0
    x = (x * 255).astype(np.uint8)
    return x


def plot_matrix(x, outfile, **kwargs):
    img = mat_to_img(x, **kwargs)
    save_image(img, outfile)


def plot_spot_masked_image(
        locs, values, mask, size, outfile=None):
    mask_indices = np.stack(np.where(mask), -1)
    shape = np.array(mask.shape)
    offset = (-1) * shape // 2
    locs = locs + offset
    img = np.full(size, np.nan)
    for loc, val in zip(locs, values):
        indices = loc + mask_indices
        img[indices[:, 0], indices[:, 1]] = val
    plot_matrix(img, outfile)


def plot_labels_3d(labs, filename=None):
    depth = labs.shape[0]
    labs = rearrange(labs, 'd h w -> (d h) w')
    img = plot_labels(labs, filename)
    img = rearrange(img, '(d h) w c -> d h w c', d=depth)
    return img


def compress_indices(indices):
    indices = indices.astype(np.int32)
    indices = np.unique(indices, axis=0)
    return indices


def plot_cells(x, masks, filename, tissue=None, boundaries=None):
    if tissue is None:
        shape = np.max([m.max(0) for m in masks], 0)
        shape = np.ceil(shape).astype(int) + 1
    else:
        shape = tissue.shape
    mat = np.zeros(shape, dtype=np.float32)
    for u, indices in zip(x, masks):
        indices = compress_indices(indices)
        mat[indices[:, 0], indices[:, 1]] = u
    if tissue is not None:
        mat[~tissue] = np.nan
    if boundaries is not None:
        for indices in boundaries:
            indices = compress_indices(indices)
            mat[indices[:, 0], indices[:, 1]] = np.nan
    plot_matrix(mat, filename)


def plot_colorbar(cmap, n_labels, filename):

    size = (200, 200)

    if n_labels is None:
        x = np.linspace(0, 1, size[0]*10)
    else:
        x = np.arange(n_labels)
        x = np.repeat(x, size[0])

    x = np.tile(x[:, np.newaxis], size[1])
    x = x.swapaxes(0, 1)

    if n_labels is None:
        plot_matrix(x, filename, cmap=cmap)
    else:
        plot_labels(x, filename, cmap=cmap)
