import sys

import numpy as np
import pandas as pd

from utils import load_pickle, save_tsv, read_lines


def get_masks(labels):
    if labels.ndim == 3:
        labels, labels_dict = flatten_labels(labels)
    else:
        uniq = np.unique(labels)
        uniq = uniq[uniq >= 0]
        labels_dict = uniq[..., np.newaxis]
    labels_uniq = np.unique(labels[labels >= 0])
    masks = [labels == lab for lab in labels_uniq]
    masks = np.array(masks)
    return masks, labels_dict


def flatten_labels(labels):
    isin = (labels >= 0).all(-1)
    flat = np.full_like(labels[..., 0], -99)
    flat[~isin] = -1
    dic, indices = np.unique(
            labels[isin], return_inverse=True, axis=0)
    flat[isin] = indices
    return flat, dic


def to_str(labels):
    labels = np.char.zfill(labels.astype(str), 2)
    labels = ['_'.join(e) for e in labels]
    return labels


def aggregate(x, masks, labels):
    groups = [x[ma] for ma in masks]
    df = pd.DataFrame([[g.size, g.mean(), g.var()] for g in groups])
    df.columns = ['count', 'mean', 'variance']
    labels = to_str(labels)
    df.index = labels
    df.index.name = 'cluster'
    return df


def aggregate_files(prefix, gene_names, masks, labels):
    for gname in gene_names:
        x = load_pickle(f'{prefix}cnts-super/{gname}.pickle')
        stats = aggregate(x, masks, labels)
        save_tsv(stats, f'{prefix}cnts-clustered/by-genes/{gname}.tsv')


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    clus = load_pickle(f'{prefix}clusters-gene/labels.pickle')
    masks, labels = get_masks(clus)
    gene_names = read_lines(prefix+'gene-names.txt')
    aggregate_files(prefix, gene_names, masks, labels)


if __name__ == '__main__':
    main()
