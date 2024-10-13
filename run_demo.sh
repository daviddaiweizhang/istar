#!/bin/bash
set -e

prefix="data/demo/"

# download demo data
./download_demo.sh $prefix
# download pretrained models
./download_checkpoints.sh
# run pipeline
./run.sh $prefix
