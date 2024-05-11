#!/bin/bash
set -e

prefix="data/markers/"

source_celltype="https://upenn.box.com/shared/static/us6015dt0bpe44w8ar6syutmbfgsl44u.tsv"
source_tls="https://upenn.box.com/shared/static/97bch259c42s8tbyr6v2iu5plz1jbkz5.txt"

target_celltype="${prefix}celltype.tsv"
target_tls="${prefix}tls.txt"

mkdir -p `dirname $target_celltype`
wget ${source_celltype} -O ${target_celltype}
wget ${source_tls} -O ${target_tls}
