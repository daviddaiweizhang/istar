import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


def reduce_dim(
        x, n_components, method='pca',
        pre_normalize=False, post_normalize=False):

    if n_components >= 1:
        n_components = int(n_components)

    isfin = np.isfinite(x).all(-1)
    if pre_normalize:
        x -= x[isfin].mean(0)
        x /= x[isfin].std(0)

    if method == 'pca':
        model = PCA(n_components=n_components)
    elif method == 'umap':
        model = UMAP(
            n_components=n_components, n_neighbors=20, min_dist=0.0,
            n_jobs=-1, random_state=0, verbose=True)
    else:
        raise ValueError(f'Method `{method}` not recognized')

    print(x[isfin].shape)
    u = model.fit_transform(x[isfin])
    print('n_components:', u.shape[-1], '/', x.shape[-1])
    if method == 'pca':
        print('pve:', model.explained_variance_ratio_.sum())

    # order components by variance
    order = np.nanvar(u, axis=0).argsort()[::-1]
    u = u[:, order]
    # make all components have variance == 1
    if post_normalize:
        u -= u.mean(0)
        u /= u.std(0)
    z = np.full(
            isfin.shape + (u.shape[-1],),
            np.nan, dtype=np.float32)
    z[isfin] = u
    return z, model
