import os
from time import time
import argparse

from einops import rearrange, reduce, repeat
import numpy as np
import skimage
import torch

from utils import load_image
from hipt_model_utils import eval_transforms
from hipt_4k import HIPT_4K
from utils import load_pickle, save_pickle, join
from image import upscale, smoothen
# from distill import distill_embeddings
from connected_components import get_largest_connected
from reduce_dim import reduce_dim


def load_mask(filename):
    mask = load_image(filename)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    factor = 16
    mask = reduce(
            mask.astype(np.float32),
            '(h0 h1) (w0 w1) -> h0 w0', 'mean',
            h1=factor, w1=factor) > 0.5
    return mask


def match_foregrounds(embs, largest_only=False):
    print('Matching foregrounds...')
    t0 = time()
    channels = np.concatenate(list(embs.values()))
    mask = np.isfinite(channels).all(0)
    if largest_only:
        mask = get_largest_connected(mask)
    for group, channels in embs.items():
        for chan in channels:
            chan[~mask] = np.nan
    print(int(time() - t0), 'sec')


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    x = np.pad(
            x,
            (
                (0, shape_ext[0] - x.shape[0]),
                (0, shape_ext[1] - x.shape[1]),
                (0, 0)),
            mode='edge')
    tiles_shape = np.array(x.shape[:2]) // patch_size
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> h1 w1 h w c',
    #         h=patch_size, w=patch_size)
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> (h1 w1) h w c',
    #         h=patch_size, w=patch_size)
    tiles = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size  # TODO: change to patch_size[0]
        b0 = a0 + patch_size  # TODO: change to patch_size[0]
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size  # TODO: change to patch_size[1]
            b1 = a1 + patch_size  # TODO: change to patch_size[1]
            tiles.append(x[a0:b0, a1:b1])

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    return tiles, shapes


def get_data(prefix):
    img = load_image(f'{prefix}he.jpg')
    return img


def get_embeddings_sub(model, x):
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def get_embeddings_cls(model, x):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        __, x_sub4k = model.forward_all4k(x[None])
    x_sub4k = x_sub4k.cpu().detach().numpy()
    x_sub4k = x_sub4k[0].transpose(1, 2, 0)
    return x_sub4k


def get_embeddings(img, pretrained=True, device='cuda'):
    '''
    Extract embeddings from histology tiles
    Args:
        tiles: Histology image tiles.
            Shape: (N, H, W, C).
            `H` and `W` are both divisible by 256.
            Channels `C` include R, G, B, foreground mask.
    Returns:
        emb_cls: Embeddings of (256 x 256)-sized patches
            Shape: (H/256, W/256, 384)
        emb_sub: Embeddings of (16 x 16)-sized patches
            Shape: (H/16, W/16, 384)
    '''
    print('Extracting embeddings...')
    t0 = time()

    tile_size = 4096
    tiles, shapes = patchify(img, patch_size=tile_size)

    model256_path, model4k_path = None, None
    if pretrained:
        model256_path = 'checkpoints/vit256_small_dino.pth'
        model4k_path = 'checkpoints/vit4k_xs_dino.pth'
    model = HIPT_4K(
            model256_path=model256_path,
            model4k_path=model4k_path,
            device256=device, device4k=device)
    model.eval()
    patch_size = (256, 256)
    subpatch_size = (16, 16)
    n_subpatches = tuple(
            a // b for a, b in zip(patch_size, subpatch_size))

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        if i % 10 == 0:
            print('tile', i, '/', len(tiles))
        x_mid, x_sub = get_embeddings_sub(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)
    del tiles
    torch.cuda.empty_cache()
    emb_mid = rearrange(
            emb_mid, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=shapes['tiles'][0], w1=shapes['tiles'][1])

    emb_cls = get_embeddings_cls(model, emb_mid)
    del emb_mid, model
    torch.cuda.empty_cache()

    shape_orig = np.array(shapes['original']) // subpatch_size

    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
                np.array([e[..., i] for e in emb_sub]),
                '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
                h1=shapes['tiles'][0], w1=shapes['tiles'][1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_sub.append(chan)
    del emb_sub

    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
                np.array([e[..., i] for e in emb_cls]),
                'h12 w12 -> (h12 h3) (w12 w3)',
                h3=n_subpatches[0], w3=n_subpatches[1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_cls.append(chan)
    del emb_cls

    print(int(time() - t0), 'sec')

    return chans_cls, chans_sub


def get_embeddings_shift(
        img, margin=256, stride=64,
        pretrained=True, device='cuda'):
    # margin: margin for shifting. Divisble by 256
    # stride: stride for shifting. Divides `margin`.
    factor = 16  # scaling factor between cls and sub. Fixed
    shape_emb = np.array(img.shape[:2]) // factor
    chans_cls = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(192)]
    chans_sub = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(384)]
    start_list = list(range(0, margin, stride))
    n_reps = 0
    for start0 in start_list:
        for start1 in start_list:
            print(f'shift {start0}/{margin}, {start1}/{margin}')
            t0 = time()
            stop0, stop1 = -margin+start0, -margin+start1
            im = img[start0:stop0, start1:stop1]
            cls, sub = get_embeddings(
                    im, pretrained=pretrained, device=device)
            del im
            sta0, sta1 = start0 // factor, start1 // factor
            sto0, sto1 = stop0 // factor, stop1 // factor
            for i in range(len(chans_cls)):
                chans_cls[i][sta0:sto0, sta1:sto1] += cls[i]
            del cls
            for i in range(len(chans_sub)):
                chans_sub[i][sta0:sto0, sta1:sto1] += sub[i]
            del sub
            n_reps += 1
            print(int(time() - t0), 'sec')

    mar = margin // factor
    for chan in chans_cls:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0
    for chan in chans_sub:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0

    return chans_cls, chans_sub


def reshape_embeddings(emb_cls, emb_sub, tiles_shape):
    # emb_cls = emb_cls.reshape(tiles_shape + emb_cls.shape[1:])
    # emb_sub = emb_sub.reshape(tiles_shape + emb_sub.shape[1:])
    emb_cls = rearrange(
            emb_cls, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=tiles_shape[0], w1=tiles_shape[1])
    # emb_sub = rearrange(
    #         emb_sub, 'h1 w1 h2 w2 h3 w3 k -> (h1 h2 h3) (w1 w2 w3) k')
    return emb_cls, emb_sub


def transpose_channels(x):
    return [x[..., i] for i in range(x.shape[-1])]


def transpose_embeddings(embs, groups=None):
    if groups is None:
        groups = embs.keys()
    out = {}
    for key, chans in embs.items():
        if key in groups:
            out[key] = transpose_channels(chans)
        else:
            out[key] = chans
    return out


def match_resolutions(embs, target_shape, groups=None):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            print(f'Matching {grp} embedding resolutions...')
            t0 = time()
            em = [
                    upscale(im[..., np.newaxis], target_shape)[..., 0]
                    for im in em]
            print(int(time() - t0), 'sec')
        out[grp] = em

    return out


def combine_embs(embs):
    embs_new = {}
    for key, channels in embs.items():
        channels = [c - np.nanmean(c) for c in channels]
        variances = [np.nanmean(c**2) for c in channels]
        std = np.sum(variances)**0.5
        channels = [c / std for c in channels]
        embs_new[key] = channels
    embs_new = join(list(embs_new.values()))
    return embs_new


def rearrange_slide(tiles, shape):
    tiles = rearrange(
            tiles, '(h1 w1) h w c -> (h1 h) (w1 w) c',
            h1=shape[0], w1=shape[1])
    return tiles


def downscale(x, factors):
    x = reduce(
            x, '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factors[0], w=factors[1])
    return x


def downscale_embedding(emb_dict, factor, groups=None):
    if groups is None:
        groups = emb_dict.keys()
    print('Downscaling slides...')
    t0 = time()
    factor = (factor, factor)
    y = {}
    for key, channel_list in emb_dict.items():
        if key in groups:
            channel_list_new = [
                downscale(channel[..., np.newaxis], factor)[..., 0]
                for channel in channel_list]
        else:
            channel_list_new = channel_list
        y[key] = channel_list_new
    print(int(time() - t0), 'sec')
    return y


def save_embeddings(x, outfile):
    print('Saving embeddings...')
    t0 = time()
    save_pickle(x, outfile)
    print(int(time() - t0), 'sec')
    print('Embeddings saved to', outfile)


def reduce_embs_dim(
        embs, n_components, method='pca', balance=False,
        groups=None):
    print(f'Reducing dimension of embeddings using {method}...')

    if groups is None:
        groups = embs.keys()

    embs_dict = {}
    models_dict = {}
    for grp, em in embs.items():
        if grp in groups:
            t0 = time()
            em, mod = reduce_dim(
                    em, n_components=n_components, method=method)
        else:
            mod = None
        embs_dict[grp] = em
        models_dict[grp] = mod
        print('runtime:', int(time() - t0), 'sec')

    return embs_dict, models_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--reduction-method', type=str, default=None)
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--smoothen-method', type=str, default='cv')
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('--no-shift', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args


# TODO: try more sophisticated methods in HistomicsTK
def color_deconvolution(x):
    mask = np.isfinite(x)
    x[~mask] = 0.0
    x = (x * 255).astype(np.uint8)
    x = skimage.color.rgb2hed(x)
    x[~mask] = np.nan
    return x


def recolor(tiles):
    h1, w1 = tiles.shape[:2]  # number of tiles
    h2, w2 = 16, 16  # number of patches

    tiles = rearrange(
            tiles,
            'h1 w1 (h2 h) (w2 w) c -> '
            '(h1 w1 h2 w2) h w c',
            h2=h2, w2=w2)
    tiles = [color_deconvolution(t) for t in tiles]
    tiles = rearrange(
            tiles,
            '(h1 w1 h2 w2) (h w) c ->'
            'h1 w1 (h2 h) (w2 w) c',
            h1=h1, w1=w1, h2=h2, w2=w2)
    return tiles


def smoothen_embeddings(
        embs, size, kernel,
        method='cv', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [
                        smoothen(
                            c[..., np.newaxis], size=size,
                            kernel=kernel, backend=method,
                            device=device)[..., 0]
                        for c in em]
            else:
                smoothened = smoothen(em, size, method, device=device)
        else:
            smoothened = em
        out[grp] = smoothened
    return out


def adjust_weights(embs, weights=None):
    print('Adjusting weights...')
    t0 = time()
    if weights is None:
        weights = {grp: 1.0 for grp in embs.keys()}
    for grp in embs.keys():
        channels = embs[grp]
        wt = weights[grp]
        means = np.array([np.nanmean(chan) for chan in channels])
        std = np.sum([np.nanvar(chan) for chan in channels])**0.5
        for chan, me in zip(channels, means):
            chan[:] -= me
            chan[:] /= std
            chan[:] *= wt**0.5
    print(int(time() - t0), 'sec')


def quantize(x, labels, hardness=0.5):
    y = np.full_like(x, np.nan)
    for lab in np.unique(labels):
        isin = lab == labels
        y[isin] = x[isin].mean(0) * hardness + x[isin] * (1 - hardness)
    return y


def main():
    args = get_args()

    np.random.seed(0)
    torch.manual_seed(0)

    # load data
    wsi = get_data(prefix=args.prefix)

    if args.use_cache:
        cache_file = args.prefix + 'embeddings-hist-raw.pickle'
    if args.use_cache and os.path.exists(cache_file):
        embs = load_pickle(cache_file)
    else:
        # extract HIPT embeddings
        if not args.no_shift:
            emb_cls, emb_sub = get_embeddings_shift(
                    wsi, pretrained=(not args.random_weights),
                    device=args.device)
        else:
            emb_cls, emb_sub = get_embeddings(
                    wsi, pretrained=(not args.random_weights),
                    device=args.device)
        embs = dict(cls=emb_cls, sub=emb_sub)
    if args.use_cache:
        save_embeddings(embs, cache_file)

    embs['rgb'] = np.stack([
            reduce(
                wsi[..., i].astype(np.float16) / 255.0,
                '(h1 h) (w1 w) -> h1 w1', 'mean',
                h=16, w=16).astype(np.float32)
            for i in range(3)])
    del wsi

    # smoothen embeddings
    if args.smoothen_method is not None:
        print('Smoothening cls embeddings...')
        t0 = time()
        embs = smoothen_embeddings(
                embs, size=16, kernel='uniform', groups=['cls'],
                method=args.smoothen_method,
                device=args.device)
        print('runtime:', int(time()-t0))

        print('Smoothening sub embeddings...')
        t0 = time()
        embs = smoothen_embeddings(
                embs, size=4, kernel='uniform', groups=['sub'],
                method=args.smoothen_method,
                device=args.device)
        print('runtime:', int(time()-t0))

    # reduce embedding dimension
    if args.reduction_method is not None:
        embs, reducers = reduce_embs_dim(
                embs, n_components=args.n_components,
                method=args.reduction_method, balance=False,
                groups=['cls', 'sub'])
        save_pickle(reducers, args.prefix+'reducers.pickle')

    save_embeddings(embs, args.prefix + 'embeddings-hist.pickle')


if __name__ == '__main__':
    main()
