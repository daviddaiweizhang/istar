#!/bin/bash
set -e

source_256="https://upenn.box.com/shared/static/p0hc12l1bpu5c7fzieotv1d6592btv1l.pth"
source_4k="https://upenn.box.com/shared/static/8qayhxzmdjpcr5loi88xtkfbqomag8a9.pth"
target_256="checkpoints/vit256_small_dino.pth"
target_4k="checkpoints/vit4k_xs_dino.pth"

mkdir -p checkpoints
wget ${source_256} -O ${target_256}
wget ${source_4k} -O ${target_4k}
