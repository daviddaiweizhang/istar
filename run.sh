#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis
n_genes=1000  # number of most variable genes to impute

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
python rescale.py ${prefix} --image
python preprocess.py ${prefix} --image

# extract histology features
python extract_features.py ${prefix} --device=${device}
# # If you want to retun model, you need to delete the existing results:
# rm ${prefix}embeddings-hist-raw.pickle

# auto detect tissue mask
# If you have a user-defined tissue mask, put it at `${prefix}mask-raw.png` and comment out the line below
python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}mask-small.png

# # segment image by histology features
# python cluster.py --mask=${prefix}mask-small.png --n-clusters=10 ${prefix}embeddings-hist.pickle ${prefix}clusters-hist/
# # # segment image by histology features without tissue mask
# # python cluster.py ${prefix}embeddings-hist.pickle ${prefix}clusters-hist/unmasked/

# select most highly variable genes to predict
# If you have a user-defined list of genes, put it at `${prefix}gene-names.txt` and comment out the line below
python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

# predict super-resolution gene expression
# rescale coordinates and spot radius
python rescale.py ${prefix} --locs --radius

# train gene expression prediction model and predict at super-resolution
python impute.py ${prefix} --epochs=400 --device=${device}  # train model from scratch
# # If you want to retrain model, you need to delete the existing model:
# rm -r ${prefix}states

# visualize imputed gene expression
python plot_imputed.py ${prefix}

# segment image by gene features
python cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=10 --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/
# # segment image without tissue mask
# python cluster.py --filter-size=8 --min-cluster-size=20 ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unmasked/
# # segment image without spatial smoothing
# python cluster.py --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unsmoothed/
# python cluster.py ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unsmoothed/unmasked/

# differential analysis by clusters
python aggregate_imputed.py ${prefix}
python reorganize_imputed.py ${prefix}
python differential.py ${prefix}

# visualize spot-level gene expression data
python plot_spots.py ${prefix}

# # cell type inference
# # see data/markers/cell-type-template.tsv for an example of a cell type reference panel
# python pixannot.py ${prefix} data/markers/cell-type.tsv ${prefix}markers/cell-type/
# cp -r ${prefix}markers/cell-type/threshold010/* ${prefix}markers/cell-type/
# python enrich.py ${prefix}clusters-gene/ ${prefix}markers/cell-type/

# # user-defined tissue structure signature scores
# # see data/markers/signature-score-template.tsv for an example of a signature score reference panel
# python marker_score.py ${prefix} data/markers/signature-score.txt ${prefix}markers/signature-score
