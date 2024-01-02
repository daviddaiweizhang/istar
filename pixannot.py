import sys
import re

import numpy as np

from utils import save_pickle, read_lines, write_lines, load_tsv
from visual import plot_matrix, plot_labels, plot_label_masks
from marker_score import get_marker_score


def adjust_temperature(probs, temperature):
    logits = np.log(probs)
    logits /= temperature
    probs = np.exp(logits)
    probs = probs / probs.sum(-1, keepdims=True)
    return probs


def sample_from_scores(x, temperature=0.05):
    probs_raw = x / x.sum(-1, keepdims=True)
    probs = adjust_temperature(probs_raw, temperature=temperature)
    z = np.random.rand(*probs.shape[:-1], 1)
    threshs = np.cumsum(probs, -1)
    labels = (z > threshs).sum(-1)
    return labels


def get_scores(prefix, marker_file):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    df = load_tsv(marker_file, index=False)
    df = df[['gene', 'label']]
    labels = np.sort(df['label'].unique()).tolist()
    scores = []
    for lab in labels:
        isin = (df['label'] == lab).to_numpy()
        gene_names = df['gene'][isin].to_numpy()
        sco = get_marker_score(prefix, gene_names)
        scores.append(sco)
    scores = np.stack(scores, -1)
    return scores, labels


def predict(scores, sample=False):
    mask = np.isfinite(scores).all(-1)
    if sample:
        labels = sample_from_scores(scores, temperature=0.05)
        labels[~mask] = -1
    else:
        labels = np.full(mask.shape, -1)
        labels[mask] = scores[mask].argmax(-1)
    return labels


def clean(s):
    s = re.sub('[^0-9a-zA-Z]+', '-', s)
    s = s.lower()
    return s


def plot_annot(labels, confidence, threshold, label_names, prefix):

    labels = labels.copy()

    # treat low-confidence predictions as unclassified
    labels[labels >= 0] += 1
    labels[confidence < threshold] = 0
    lab_names = ['Unclassified'] + label_names

    write_lines(lab_names, f'{prefix}label-names.txt')
    save_pickle(labels, f'{prefix}labels.pickle')
    plot_labels(
            labels, f'{prefix}labels.png',
            white_background=True,
            cmap='tab10')
    plot_labels(
            labels, f'{prefix}labels-altcmap.png',
            white_background=True,
            cmap='Set3')
    lab_names_clean = [clean(lname) for lname in lab_names]
    plot_label_masks(
            labels, f'{prefix}masks/',
            names=lab_names_clean,
            white_background=True)


def main():

    np.random.seed(0)

    prefix_inp = sys.argv[1]  # e.g. data/her2st/H123/
    marker_file = sys.argv[2]  # e.g. data/markers/celltype.tsv
    prefix_out = sys.argv[3]  # e.g. data/her2st/H123/cell-types/

    scores, lab_names = get_scores(prefix_inp, marker_file)

    for x, lname in zip(scores.transpose(2, 0, 1), lab_names):
        plot_matrix(
                x, f'{prefix_out}scores/{clean(lname)}.png',
                white_background=True)

    confidence = scores.max(-1)
    plot_matrix(
            confidence, f'{prefix_out}confidence.png',
            white_background=True)

    labels = predict(scores)

    for threshold in [0.01, 0.05, 0.10, 0.20]:
        plot_annot(
                labels, confidence, threshold, lab_names,
                f'{prefix_out}threshold{int(threshold*1000):03d}/')


if __name__ == '__main__':
    main()
