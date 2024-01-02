#!/bin/bash
set -e

prefix="data/markers/"

source_celltype="https://upenn.box.com/shared/static/t07qhkych3sx2ak1f19c9ptphssb1gd0.tsv"
source_tls="https://upenn.box.com/shared/static/9gfiq5yyuv9ynfgkb5df7z64kurtekjl.txt"

target_celltype="${prefix}celltype.tsv"
target_tls="${prefix}tls.txt"

mkdir -p `dirname $target_celltype`
wget ${source_celltype} -O ${target_celltype}
wget ${source_tls} -O ${target_tls}
