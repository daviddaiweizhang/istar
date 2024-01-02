import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import seaborn as sns

from utils import load_pickle, read_lines, save_tsv


def get_data(prefix0, prefix1):
    labels0 = load_pickle(prefix0+'labels.pickle')
    labels1 = load_pickle(prefix1+'labels.pickle')
    labels1_names = read_lines(prefix1+'label-names.txt')
    return labels0, labels1, labels1_names


def get_probs(labels0, labels1):
    nlabs0 = labels0.max() + 1
    nlabs1 = labels1.max() + 1
    results = np.full((nlabs0, nlabs1), np.nan)
    for l0 in range(nlabs0):
        for l1 in range(nlabs1):
            m0 = labels0 == l0
            m1 = labels1 == l1
            results[l0, l1] = (m0 * m1).sum()
    results /= np.nansum(results)
    return results


def plot_probs(df, filename, cmap='tab10'):

    font = {'size': 15}
    plt.rc('font', **font)

    cmap = plt.get_cmap(cmap)
    color = [cmap(i) for i in range(df.shape[1])]
    df = df / np.nansum(df, 1, keepdims=True)
    df.plot(kind='bar', stacked=True, color=color)
    plt.xlabel('Cluster')
    plt.ylabel('Cell type proportion')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(filename)


def plot_results(df, filename, cmap='tab10'):

    font = {'size': 15}
    plt.rc('font', **font)

    cmap = plt.get_cmap(cmap)
    color = [cmap(i) for i in range(df.shape[1])]
    df.plot(kind='bar', stacked=True, color=color)
    plt.xlabel('Cluster')
    plt.ylabel('Cell type proportion')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(filename)


def probs_to_oddsratios(df):
    x = df.to_numpy()
    oddsratios = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            not_i = [k for k in range(x.shape[0]) if k != i]
            not_j = [k for k in range(x.shape[1]) if k != j]
            a = x[i, j] * x[not_i][:, not_j].sum()
            b = x[i, not_j].sum() * x[not_i, j].sum()
            oddsratios[i, j] = a / b
    oddsratios = pd.DataFrame(oddsratios)
    oddsratios.columns = df.columns
    oddsratios.index = df.index
    return oddsratios


def plot_enrichment(df, filename):
    sns.heatmap(
            df, cmap='magma',
            annot=True, fmt='.1f', annot_kws={'fontsize': 12},
            square=True, linewidth=0.5)

    # # set x-ticks on top
    # ax.set(xlabel='', ylabel='')
    # ax.xaxis.tick_top()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(filename)


def process_oddsratios(df):
    df = df.drop(columns='Unclassified')
    x = df.to_numpy()
    threshold = 2**0.05  # avoid OR == 0.0 after rounding
    x[x < threshold] = np.nan
    x = np.log2(x)
    df[:] = x
    df = df.T
    return df


def main():
    prefix0 = sys.argv[1]  # e.g. 'data/her2st/H123/clusters-gene/'
    prefix1 = sys.argv[2]  # e.g. 'data/her2st/H123/markers/celltype/'
    labels0, labels1, labels1_names = get_data(prefix0, prefix1)

    probs = get_probs(labels0, labels1)
    probs = pd.DataFrame(probs)
    probs.columns = labels1_names
    plot_probs(
            probs,
            cmap='tab10',
            filename=prefix0+'proportions.png')
    plot_probs(
            probs,
            cmap='Set3',
            filename=prefix0+'proportions-altcmap.png')

    oddsratios = probs_to_oddsratios(probs)
    oddsratios = process_oddsratios(oddsratios)
    save_tsv(
            oddsratios, prefix0+'enrichment.csv',
            sep=',', na_rep='NA')
    plot_enrichment(oddsratios, prefix0+'enrichment.png')


if __name__ == '__main__':
    main()
