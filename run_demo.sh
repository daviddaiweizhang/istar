#!/bin/bash
set -e

prefix="data/demo/"

# download demo data
./download_demo.sh $prefix
# download reference data
./download_references.sh
# download pretrained models
./download_checkpoints.sh
# run pipeline
./run.sh $prefix
