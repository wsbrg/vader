#!/bin/bash

set -e

# Create directory for temporary files
mkdir -p tmp

# Attack dependencies
APPTAINER_TMPDIR=$(pwd)/tmp apptainer build vader.sif vader.def
APPTAINER_TMPDIR=$(pwd)/tmp apptainer build deeplab.sif deeplab.def

# Baselines
apptainer build baselines/hilgefort/hilgefort.sif baselines/hilgefort/hilgefort.def
apptainer build baselines/sabra/sabra.sif baselines/sabra/sabra.def

# Download u2net for Hilgefort baseline
# https://github.com/LeCongThuong/U2Net
curl 'https://drive.usercontent.google.com/download?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&export=download&authuser=0&confirm=t&at=AIrpjvPhYs8St6rqYrLdjWYfT-Z_%3A173670294431' -o baselines/hilgefort/u2net.pth
