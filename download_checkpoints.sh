#!/bin/bash
set -e

source_256="https://upenn.box.com/shared/static/1rxqte1zfxehjn09oh4i84qknyjvkrp0.pth"
source_4k="https://upenn.box.com/shared/static/vw7jdi4hdykl5xlk3o84napu399kvv5n.pth"
target_256="checkpoints/vit256_small_dino.pth"
target_4k="checkpoints/vit4k_xs_dino.pth"

mkdir -p checkpoints
wget ${source_256} -O ${target_256}
wget ${source_4k} -O ${target_4k}
