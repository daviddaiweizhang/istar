#!/bin/bash
set -e

prefix=$1

source_img="https://upenn.box.com/shared/static/47c9w1klf1tp2hmtyfuzymikrytmazx1.jpg"
source_cnts="https://upenn.box.com/shared/static/30l8agtu1pjjb7o8gpictekbkgl0hhu9.tsv"
source_locs="https://upenn.box.com/shared/static/ocjl50ngya4f7jd6nbyrmt446it6mo9c.tsv"
source_radius="https://upenn.box.com/shared/static/o938gji6435prj0i6vhx6hj7cvv5fjot.txt"
source_pixsize="https://upenn.box.com/shared/static/pktuthimjfqn473w7xqpptcvg6lnnyud.txt"

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
