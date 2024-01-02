import sys

import numpy as np
from einops import reduce

from utils import load_image, load_tsv, read_lines, read_string
from visual import plot_spots


# def plot_spots(cnts, locs, underground, gene_names, radius, prefix):
#     under_weight = 0.2
#     cmap = plt.get_cmap('turbo')
#     under = underground.mean(-1, keepdims=True)
#     under = np.tile(under, 3)
#     under -= under.min()
#     under /= under.max() + 1e-12
#     for k, name in enumerate(gene_names):
#         x = cnts[:, k]
#         x = x - x.min()
#         x = x / (x.max() + 1e-12)
#         img = under * under_weight
#         for u, ij in zip(x, locs):
#             lower = np.clip(ij - radius, 0, None)
#             upper = np.clip(ij + radius, None, img.shape[:2])
#             color = np.array(cmap(u)[:3]) * (1 - under_weight)
#             img[lower[0]:upper[0], lower[1]:upper[1]] += color
#         img = (img * 255).astype(np.uint8)
#         save_image(img, f'{prefix}{name}.png')


def plot_spots_multi(
        cnts, locs, gene_names, radius, img, prefix,
        disk_mask=True):
    for i, gname in enumerate(gene_names):
        ct = cnts[:, i]
        outfile = f'{prefix}{gname}.png'
        plot_spots(
                img=img, cnts=ct, locs=locs, radius=radius,
                cmap='turbo', weight=1.0,
                disk_mask=disk_mask,
                outfile=outfile)


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    factor = 16

    infile_cnts = f'{prefix}cnts.tsv'
    infile_locs = f'{prefix}locs.tsv'
    infile_img = f'{prefix}he.jpg'
    infile_genes = f'{prefix}gene-names.txt'
    infile_radius = f'{prefix}radius.txt'

    # load data
    cnts = load_tsv(infile_cnts)
    locs = load_tsv(infile_locs)
    assert (cnts.index == locs.index).all()
    spot_radius = int(read_string(infile_radius))
    img = load_image(infile_img)

    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if img.ndim == 2:
        img = np.tile(img[..., np.newaxis], 3)

    # select genes
    gene_names = read_lines(infile_genes)
    cnts = cnts[gene_names]
    cnts = cnts.to_numpy()

    # recale image
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs /= factor
    locs = locs.round().astype(int)
    img = reduce(
            img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)

    # rescale spot
    spot_radius = np.round(spot_radius / factor).astype(int)

    # plot spot-level gene expression measurements
    plot_spots_multi(
            cnts=cnts,
            locs=locs, gene_names=gene_names,
            radius=spot_radius, disk_mask=True,
            img=img, prefix=prefix+'spots/')


if __name__ == '__main__':
    main()
