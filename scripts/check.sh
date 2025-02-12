#!/bin/bash

set -e

echo "[+] Check if u2net is available"
ls baselines/hilgefort/u2net.pth &> /dev/null

echo "[+] Check script 'src/generate/generate_bg_videos.py'"
./vader.sif python3 src/generate/generate_bg_videos.py --help &> /dev/null
echo "[+] Check script 'src/generate/generate_mp_masks'"
apptainer run --nv vader.sif python3 src/generate/generate_mp_masks.py --help &> /dev/null
echo "[+] Check script 'src/generate/generate_vbg_videos'"
./vader.sif python3 src/generate/generate_vbg_videos.py --help &> /dev/null
echo "[+] Check script 'src/attack/mp.py'"
apptainer run --nv vader.sif python3 src/attack/mp.py --help &> /dev/null
echo "[+] Check script 'src/deeplab/deeplab.py'"
apptainer run --nv --no-mount home deeplab.sif python3 src/deeplab/deeplab.py --help &> /dev/null
echo "[+] Check script 'src/gt/gt.py'"
./vader.sif python3 src/gt/gt.py --help &> /dev/null
echo "[+] Check script 'src/attack/deltae.py'"
./vader.sif python3 src/attack/deltae.py --help &> /dev/null
echo "[+] Check script 'src/attack/attack.py'"
./vader.sif python3 src/attack/attack.py --help &> /dev/null
echo "[+] Check script 'baselines/sabra/sabra.py'"
apptainer run --nv baselines/sabra/sabra.sif python3 baselines/sabra/sabra.py --help &> /dev/null
echo "[+] Check script 'baselines/hilgefort/hilgefort.py'"
apptainer run --nv baselines/hilgefort/hilgefort.sif python3 baselines/hilgefort/hilgefort.py --help &> /dev/null
echo "[+] Check script 'src/eval/deltae.py'"
./vader.sif python3 src/eval/deltae.py --help &> /dev/null
echo "[+] Check script 'src/eval/eval.py'"
./vader.sif python3 src/eval/eval.py --help &> /dev/null

echo "[*] Success!"
