import sys

import numpy as np

from utils import save_image, load_pickle
from cluster import cluster
from connected_components import relabel_small_connected
from image import crop_image


def remove_margins(embs, mar):
    for ke, va in embs.items():
        embs[ke] = [
                v[mar[0][0]:-mar[0][1], mar[1][0]:-mar[1][1]]
                for v in va]


def get_mask_embeddings(embs, mar=16, min_connected=4000):

    n_clusters = 2

    # remove margins to avoid border effects
    remove_margins(embs, ((mar, mar), (mar, mar)))

    # get features
    x = np.concatenate(list(embs.values()))

    # segment image
    labels, __ = cluster(x, n_clusters=n_clusters, method='km')
    labels = relabel_small_connected(labels, min_size=min_connected)

    # select cluster for foreground
    rgb = np.stack(embs['rgb'], -1)
    i_foreground = np.argmax([
        rgb[labels == i].std() for i in range(n_clusters)])
    mask = labels == i_foreground

    # restore margins
    extent = [(-mar, s+mar) for s in mask.shape]
    mask = crop_image(
            mask, extent,
            mode='constant', constant_values=mask.min())

    return mask


def main():

    inpfile = sys.argv[1]
    outfile = sys.argv[2]

    embs = load_pickle(inpfile)
    mask = get_mask_embeddings(embs)
    save_image(mask, outfile)


if __name__ == '__main__':
    main()
