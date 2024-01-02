import sys

import numpy as np
import pandas as pd
from scipy.stats import norm as norm_rv

from utils import load_tsv, save_tsv


def aggregate(mean, variance, count, specificity=0.02, n_top=50):

    assert (mean.index == variance.index).all()
    assert (mean.index == count.index).all()
    assert (mean.columns == variance.columns).all()
    assert (mean.columns == count.columns).all()

    gene_names = mean.columns.to_numpy()
    cluster_names = mean.index.to_numpy()
    mean = mean.to_numpy()
    vari = variance.to_numpy()
    count = count.to_numpy()

    weight = count / count.sum(0)
    vari_global = (vari * weight).sum(0)
    mean /= vari_global**0.5
    vari /= vari_global + 1e-12

    n_clusters = len(cluster_names)
    results = dict(
            mean_interior=[], mean_exterior=[],
            variance_interior=[], variance_exterior=[],
            count_interior=[], count_exterior=[])
    for i in range(n_clusters):
        isin = np.arange(n_clusters) == i
        mean_in = mean[isin].flatten()
        vari_in = vari[isin].flatten()
        count_in = count[isin].flatten()
        mean_out = (mean[~isin] * weight[~isin]).sum(0)
        vari_out = (
                (mean[~isin]**2 * weight[~isin]).sum(0)
                - mean_out**2)
        count_out = count[~isin].sum(0)
        results['mean_interior'].append(mean_in)
        results['mean_exterior'].append(mean_out)
        results['variance_interior'].append(vari_in)
        results['variance_exterior'].append(vari_out)
        results['count_interior'].append(count_in)
        results['count_exterior'].append(count_out)

    for key in results.keys():
        df = pd.DataFrame(np.array(results[key]))
        df.index = cluster_names
        df.index.name = 'cluster'
        df.columns = gene_names
        results[key] = df

    return results


def two_sample_test(mean, variance, count, mode):
    pval = mean[0].copy()
    pval[:] = np.nan
    mean = np.stack(mean)
    variance = np.stack(variance)
    count = np.stack(count)
    std = (variance / count).sum(0)**0.5
    z = (mean[1] - mean[0]) / (std + 1e-12)
    if mode == 'positive-sided':
        p = norm_rv.sf(z)
    elif mode == 'negative-sided':
        p = norm_rv.cdf(z)
    elif mode == 'two-sided':
        p = norm_rv.sf(np.abs(z)) * 2
    else:
        raise ValueError('mode not recognized')
    pval[:] = p
    return pval


def save_results(results, prefix, sort_key=None):
    cluster_names = results[list(results.keys())[0]].index.to_list()
    for cname in cluster_names:
        rslt = pd.DataFrame(
                {key: val.loc[cname] for key, val in results.items()})
        rslt.index.name = 'gene'
        if sort_key is not None:
            rslt = rslt.sort_values(sort_key, ascending=False)
            first_column = rslt.pop(sort_key) 
            rslt.insert(0, sort_key, first_column)
        save_tsv(rslt, f'{prefix}contrast/by-clusters/cluster-{cname}.tsv')
    print(
            'Top overexpressed genes for each cluster saved to '
            f'{prefix}contrast/by-clusters/')
    for key, val in results.items():
        save_tsv(val, f'{prefix}contrast/by-metrics/{key}.tsv')


def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    prefix = f'{prefix}cnts-clustered/by-clusters/'

    x_mean = load_tsv(f'{prefix}mean.tsv')
    x_vari = load_tsv(f'{prefix}variance.tsv')
    x_count = load_tsv(f'{prefix}count.tsv')

    results = aggregate(x_mean, x_vari, x_count)
    results['fold_change'] = (
            results['mean_interior'] / results['mean_exterior'])
    results['pvalue_positive_sided'] = two_sample_test(
                mean=(
                    results['mean_exterior'],
                    results['mean_interior']),
                variance=(
                    results['variance_exterior'],
                    results['variance_interior']),
                count=(
                    results['count_exterior'],
                    results['count_interior']),
                mode='positive-sided')
    results['variance_raw'] = x_vari
    save_results(results, prefix, sort_key='fold_change')


if __name__ == '__main__':
    main()
