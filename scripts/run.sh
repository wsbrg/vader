#!/bin/bash

set -e

mkdir -p tmp

 echo "[*] Generate 'real' videos from green screen recordings"
./vader.sif python3 src/generate/generate_bg_videos.py -d data -t tmp

echo "[*] Generate mediapipe masks to replace the background with a virtual one"
apptainer run --nv vader.sif python3 src/generate/generate_mp_masks.py -d data

echo "[*] Generate zoom masks to replace the background with virtual one: Use Zoom 5.9.3 and AlphaDump on Windows PC"
echo "[!] The AlphaDump tool for extracting masks from Zoom is not published due to restrictions in Zoom's terms and conditions"
echo "[!] Download pre-extracted Zoom masks for the example video instead"
tar xzf data/zoom/bg_zoom_masks.tar.gz -C data/data

echo "[*] Create virtual background replaced videos for zoom"
./vader.sif python3 src/generate/generate_vbg_videos.py -d data --tool zoom -t tmp

echo "[*] Create virtual background replaced videos for mediapipe"
./vader.sif python3 src/generate/generate_vbg_videos.py -d data --tool mp -t tmp

echo "[*] Create zoom attack masks for the zoom-vbg-videos using Zoom 5.9.3 and AlphaDump on Windows PC"
echo "[!] The AlphaDump tool for extracting masks from Zoom is not published due to restrictions in Zoom's terms and conditions"
echo "[!] Download pre-extracted Zoom masks for the example video instead"
tar xzf data/zoom/vbg_zoom_masks.tar.gz -C data/data

echo "[*] Create mp attack masks for the mp-vbg-videos"
apptainer run --nv vader.sif python3 src/attack/mp.py -d data

echo "[*] Create deeplab attack masks for every vbg video"
apptainer run --nv --no-mount home deeplab.sif python3 src/deeplab/deeplab.py -d data

echo "[*] Create leak map (ground-truths)"
./vader.sif python3 src/gt/gt.py -d data/ -p 8

echo "[*] Create vbg deltae-cache for attack"
./vader.sif python3 src/attack/deltae.py -d data

echo "[*] Conduct attack"
./vader.sif python3 src/attack/attack.py -d data

echo "[*] Run Sabra baseline attack"
apptainer run --nv baselines/sabra/sabra.sif python3 baselines/sabra/sabra.py -d data/

echo "[*] Run Hilgefort baseline attack"
apptainer run --nv baselines/hilgefort/hilgefort.sif python3 baselines/hilgefort/hilgefort.py -d data/ -u baselines/hilgefort/u2net.pth -t tmp/

echo "[*] Create deltae scores for evaluation"
./vader.sif python3 src/eval/deltae.py -d data

echo "[*] Conduct evaluation"
./vader.sif python3 src/eval/eval.py -d data --gesture interview --attack vader
./vader.sif python3 src/eval/eval.py -d data --gesture interview --attack sabra
./vader.sif python3 src/eval/eval.py -d data --gesture interview --attack hilgefort
