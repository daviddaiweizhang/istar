import numpy as np
from scipy.ndimage import label as label_connected

from utils import sort_labels, get_most_frequent


def get_largest_connected(labels):
    labels = label_connected(labels)[0]
    labels -= 1
    labels = sort_labels(labels)[0]
    labels = labels == 0
    return labels


def get_adjacent(ind):
    # return the eight adjacent indices
    adj = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij')
    adj = np.stack(adj, -1)
    adj = adj.reshape(-1, adj.shape[-1])
    adj = adj[(adj != 0).any(-1)]
    adj += ind
    return adj


def split_by_connected_size_single(labels, min_size):
    # return labels of small and large connected components
    # labels are binary
    labels = label_connected(labels)[0]
    labels -= 1
    labels = sort_labels(labels)[0]
    counts = np.unique(labels[labels >= 0], return_counts=True)[1]
    cut = np.sum(counts >= min_size)
    small = labels - cut
    small[small < 0] = -1
    large = labels.copy()
    large[labels >= cut] = -1
    return small, large


def split_by_connected_size(labels, min_size):
    # return labels of small and large connected components
    # labels can be multi-categorical
    labs_uniq = np.unique(labels[labels >= 0])
    small = np.full_like(labels, -1)
    large = np.full_like(labels, -1)
    for lab in labs_uniq:
        isin = labels == lab
        sma, lar = split_by_connected_size_single(isin, min_size)
        issma = sma >= 0
        islar = lar >= 0
        small[issma] = sma[issma] + small.max() + 1
        large[islar] = lar[islar] + large.max() + 1
    return small, large


def relabel_small_connected(labels, min_size):
    # reassign labels of small connected components
    labels = labels.copy()
    small, __ = split_by_connected_size(labels, min_size)
    small = sort_labels(small, descending=False)[0]
    small_uniq = np.unique(small[small >= 0])
    lab_na = min(-1, labels.min() - 1)
    for lab_small in small_uniq:

        isin = small == lab_small
        lab = labels[isin][0]

        # find adjacent labels
        indices = np.stack(np.where(isin), -1)
        labs_adj = []
        labs_small_adj = []
        for ind in indices:
            adj = get_adjacent(ind)
            is_within = np.logical_and(
                    (adj < labels.shape).all(-1),
                    (adj >= 0).all(-1))
            adj[~is_within] = 0  # dummy index for out-of-bound
            la = labels[adj[:, 0], adj[:, 1]]
            lsa = small[adj[:, 0], adj[:, 1]]
            la[~is_within] = lab_na
            lsa[~is_within] = lab_na
            labs_adj.append(la)
            labs_small_adj.append(lsa)
        labs_adj = np.stack(labs_adj)
        labs_small_adj = np.stack(labs_small_adj)
        # eliminate background and identical labels
        is_other = (labs_adj >= 0) * (labs_adj != lab)
        if is_other.any():
            # find most frequent adjacent labels
            lab_new = get_most_frequent(labs_adj[is_other])
            # get location of new label
            i_new, i_adj_new = np.stack(
                    np.where(labs_adj == lab_new), -1)[0]
            ind_new = get_adjacent(indices[i_new])[i_adj_new]
            # update small components
            lab_small_new = small[ind_new[0], ind_new[1]]
        else:
            lab_new = lab
            lab_small_new = lab_small
        # relabel to most frequent neighboring label
        labels[isin] = lab_new
        small[isin] = lab_small_new

    return labels


def cluster_connected(labels):
    # subcluster labels by connectedness
    labels = labels.copy()
    isfg = labels >= 0
    labels_sub = np.full_like(labels, -1)
    labels_sub[~isfg] = labels[~isfg]

    labs_uniq = np.unique(labels[isfg])
    for lab in labs_uniq:
        isin = labels == lab
        sublabs = label_connected(isin)[0] - 1
        sublabs = sort_labels(sublabs)[0]
        labels_sub[isin] = sublabs[isin]

    labels_sub[~isfg] = labels[~isfg]
    return labels_sub
