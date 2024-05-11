#!/bin/bash
set -e

prefix=$1

source_img="https://upenn.box.com/shared/static/yya0lvlur8aase29hvy630jd06r64tdn.jpg"
source_cnts="https://upenn.box.com/shared/static/kaoo8j31dx5lupyz8dctay7p5x3exqsa.tsv"
source_locs="https://upenn.box.com/shared/static/7nbnorlr2h6tkeyghjibqitztezkadwh.tsv"
source_radius="https://upenn.box.com/shared/static/a8655bmb02q9cqegndnwhcb5r0mqqphi.txt"
source_pixsize="https://upenn.box.com/shared/static/1stmq5ly6iqnljt0uq8rotlki5q8sjfs.txt"

target_img="${prefix}he-raw.jpg"
target_cnts="${prefix}cnts.tsv"
target_locs="${prefix}locs-raw.tsv"
target_radius="${prefix}radius-raw.txt"
target_pixsize="${prefix}pixel-size-raw.txt"

mkdir -p `dirname $target_img`
wget ${source_img} -O ${target_img}
wget ${source_cnts} -O ${target_cnts}
wget ${source_locs} -O ${target_locs}
wget ${source_radius} -O ${target_radius}
wget ${source_pixsize} -O ${target_pixsize}
