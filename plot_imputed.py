import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import load_pickle, save_image, read_lines, load_image
# from visual import cmap_turbo_truncated


def plot_super(
        x, outfile, underground=None, truncate=None):

    x = x.copy()
    mask = np.isfinite(x)

    if truncate is not None:
        x -= np.nanmean(x)
        x /= np.nanstd(x) + 1e-12
        x = np.clip(x, truncate[0], truncate[1])

    x -= np.nanmin(x)
    x /= np.nanmax(x) + 1e-12

    cmap = plt.get_cmap('turbo')
    # cmap = cmap_turbo_truncated
    if underground is not None:
        under = underground.mean(-1, keepdims=True)
        under -= under.min()
        under /= under.max() + 1e-12

    img = cmap(x)[..., :3]
    if underground is not None:
        img = img * 0.5 + under * 0.5
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)


def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    gene_names = read_lines(f'{prefix}gene-names.txt')
    mask = load_image(f'{prefix}mask-small.png') > 0

    for gn in gene_names:
        cnts = load_pickle(f'{prefix}cnts-super/{gn}.pickle')
        cnts[~mask] = np.nan
        plot_super(cnts, f'{prefix}cnts-super-plots/{gn}.png')


if __name__ == '__main__':
    main()
