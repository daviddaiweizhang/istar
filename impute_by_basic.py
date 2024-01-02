import sys
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from utils import (
        load_tsv, load_pickle, save_pickle, load_image, save_image,
        read_lines)
from image import smoothen


class NearestNeighborsRegressor():

    def __init__(self, n_neighbors, weights):
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        self.weights = weights

    def fit(self, x):
        self.nbrs.fit(x)

    def predict_x(self, x):
        # get neighbor indices and weights
        distances, indices = self.nbrs.kneighbors(x)
        if self.weights == 'uniform':
            wts = np.ones_like(distances)
        elif self.weights == 'distance':
            wts = 1.0 / (distances + 1e-12)
        else:
            raise ValueError('Weight function not recognized')
        wts /= wts.sum(1, keepdims=True)
        self.wts = wts
        self.indices = indices

    def predict_y(self, y):
        assert y.ndim == 2
        y_neighbors = y[self.indices]
        y_mean = (y_neighbors * self.wts[..., np.newaxis]).sum(1)
        y_diffsq = (y_neighbors - y_mean[..., np.newaxis, :])**2
        y_variance = (y_diffsq * self.wts[..., np.newaxis]).sum(1)
        return y_mean, y_variance


# def draw_overlay(locs, embs, radius, outfile):
#     em = embs[..., :3]
#     em -= np.nanmin(em)
#     em /= np.nanmax(em)
#     save_image(
#             draw_spots(
#                 locs,
#                 (em*255).astype(np.uint8),
#                 rad=radius, color=128),
#             outfile)


def log_normal(mean, variance):
    mean_new = np.exp(mean + variance * 0.5)
    variance_new = (
            (np.exp(variance) - 1)
            * np.exp(mean * 2 + variance))
    return mean_new, variance_new


def impute_by_neighbors(
        y_train, x_train, x_test, prefix,
        n_neighbors=5, weights='uniform'):
    y_train = y_train.astype(np.float32)

    model = NearestNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights)
    model.fit(x=x_train)
    mask = np.isfinite(x_test).all(-1)
    model.predict_x(x_test[mask])

    for name, y_tra in y_train.items():
        y_tra = y_tra.to_numpy()
        y_mea, y_var = model.predict_y(y_tra[..., np.newaxis])
        y_mea_arr = np.full(
                x_test.shape[:-1], np.nan, dtype=y_tra.dtype)
        y_var_arr = np.full(
                x_test.shape[:-1], np.nan, dtype=y_tra.dtype)
        y_mea_arr[mask] = y_mea[..., 0]
        y_var_arr[mask] = y_var[..., 0]
        save_pickle(y_mea_arr, f'{prefix}mean/{name}.pickle')
        save_pickle(y_var_arr, f'{prefix}variance/{name}.pickle')


def impute_by_neural(y_train, x_train, x_test, prefix, **kwargs):
    model = MLPRegressor(
            hidden_layer_sizes=(128, 128, 128, 128), activation='relu',
            learning_rate='adaptive', learning_rate_init=1e-3,
            batch_size=100, max_iter=10000, tol=1e-6,
            alpha=1e-2,
            random_state=0, verbose=True)

    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()

    names = y_train.columns
    y_train = y_train.to_numpy()
    y_train = y_train.astype(np.float32)

    x_mean = x_train.mean(0)
    x_std = x_train.std(0)
    x_train -= x_mean
    x_train /= x_std + 1e-12

    y_min = y_train.min(0)
    y_max = y_train.max(0)
    y_train -= y_min
    y_train /= (y_max - y_min) + 1e-12

    model.fit(x_train, y_train)

    x_test = x_test - x_mean
    x_test = x_test / (x_std + 1e-12)
    mask = np.isfinite(x_test).all(-1)
    y_test = model.predict(x_test[mask])
    y_test = np.clip(y_test, 0, 1)
    # threshold = 0.1
    # y_test[y_test < threshold] = 0.0

    y_test *= y_max - y_min
    y_test += y_min

    y_test_arr = np.full(
            (x_test.shape[:-1] + y_test.shape[-1:]),
            np.nan, dtype=y_test.dtype)
    y_test_arr[mask] = y_test

    idx = np.where(names == 'MS4A1')[0][0]
    aa = y_test_arr[..., idx].copy()
    aa -= np.nanmin(aa)
    aa /= np.nanmax(aa)
    cmap = plt.get_cmap('turbo')
    img = cmap(aa)[..., :3]
    save_image((img * 255).astype(np.uint8), 'a.png')

    return y_test_arr, names


def impute(y_train, x_train, x_test, prefix, method, **kwargs):

    if method == 'neighbors':
        impute_by_neighbors(y_train, x_train, x_test, prefix, **kwargs)
    elif method == 'neural':
        impute_by_neural(y_train, x_train, x_test, prefix, **kwargs)
    else:
        raise ValueError('Method not recognized.')


def get_locs(prefix, target_shape=None):

    locs = load_tsv(f'{prefix}locs.tsv')

    # change xy coordinates to ij coordinates
    locs = np.stack([locs['y'], locs['x']], -1)

    # match coordinates of embeddings and spot locations
    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    # find the nearest pixel
    locs = locs.round().astype(int)

    return locs


def get_gene_counts(prefix, reorder_genes=True):
    cnts = load_tsv(f'{prefix}cnts.tsv')
    if reorder_genes:
        order = cnts.var().to_numpy().argsort()[::-1]
        cnts = cnts.iloc[:, order]
    return cnts


def get_embeddings(prefix):
    embs = load_pickle(f'{prefix}embeddings-hist.pickle')
    embs = np.concatenate([embs['cls'], embs['sub'], embs['rgb']])
    embs = embs.transpose(1, 2, 0)
    return embs


def smoothen_batch(embs, **kwargs):
    embs_batches = np.array_split(embs, 8, axis=-1)
    embs_smooth = np.concatenate(
            [smoothen(e, **kwargs) for e in embs_batches], -1)
    return embs_smooth


def get_training_data(prefix, gene_names, spot_radius, log_counts=False):
    # get targets (gene counts)
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    # transform gene counts to log scale
    if log_counts:
        cnts = np.log(1 + cnts)

    # get features (histology embeddings)
    embs = get_embeddings(prefix)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    embs_agg = smoothen_batch(
            embs, size=spot_radius, method='cnn', fill_missing=True)
    embs_spots = embs_agg[locs[:, 0], locs[:, 1]]
    return embs_spots, cnts, embs_agg


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    spot_radius = 10

    cache_file = prefix + 'a.pickle'
    if os.path.exists(cache_file):
        embs_train, cnts_train, embs_test, embs_agg = load_pickle(cache_file)
    else:
        gene_names = read_lines(f'{prefix}gene-names.txt')
        embs_train, cnts_train, embs_agg = get_training_data(
                prefix, gene_names, spot_radius=spot_radius)
        embs_test = get_embeddings(prefix)
        save_pickle(
                (embs_train, cnts_train, embs_test, embs_agg), cache_file)

    mask = np.isfinite(embs_test).all(-1)
    embs_test[mask] = embs_agg[mask]
    # super-resolution imputation
    impute(
            y_train=cnts_train, x_train=embs_train, x_test=embs_test,
            method='neural', prefix=prefix+'cnts-super/')


if __name__ == '__main__':
    main()
