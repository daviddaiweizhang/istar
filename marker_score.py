import argparse

import numpy as np
from einops import reduce

from utils import read_lines, load_pickle, save_pickle, load_mask
from visual import plot_matrix


def compute_score(cnts, mask=None, factor=None):
    if mask is not None:
        cnts[~mask] = np.nan

    if factor is not None:
        cnts = reduce(
                cnts, '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
                h1=factor, w1=factor)

    cnts -= np.nanmin(cnts, (0, 1))
    cnts /= np.nanmax(cnts, (0, 1)) + 1e-12
    score = cnts.mean(-1)

    return score


def get_marker_score(prefix, genes_marker, threshold=None, factor=1):

    genes = read_lines(prefix+'gene-names.txt')
    mask = load_mask(prefix+'mask-small.png', verbose=False)

    gene_names = np.array(list(set(genes_marker).intersection(genes)))

    if len(gene_names) < len(genes_marker):
        print('Genes not found:')
        print(set(genes_marker).difference(set(gene_names)))

    cnts = [
            load_pickle(
                f'{prefix}cnts-super/{gname}.pickle', verbose=False)
            for gname in gene_names]
    cnts = np.stack(cnts, -1)

    if threshold is not None:
        isin = np.nanmax(cnts, (0, 1)) >= threshold
        if not isin.all():
            print('Genes that do not pass the threshold:')
            print(gene_names[~isin])
            cnts = cnts[:, :, isin]
            gene_names = gene_names[isin]

    score = compute_score(cnts, mask=mask, factor=factor)
    return score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix_inp', type=str)
    parser.add_argument('genes_marker', type=str)
    parser.add_argument('prefix_out', type=str)
    parser.add_argument('--threshold', type=float, default=1e-3)
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    # compute marker score
    genes_marker = read_lines(args.genes_marker)
    score = get_marker_score(args.prefix_inp, genes_marker, args.threshold)
    save_pickle(score, args.prefix_out+'.pickle')

    # visualize marker score
    score = np.clip(
            score, np.nanquantile(score, 0.001),
            np.nanquantile(score, 0.999))
    save_pickle(score, args.prefix_out+'.pickle')
    plot_matrix(score, args.prefix_out+'.png', white_background=True)


if __name__ == '__main__':
    main()
