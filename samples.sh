#!/bin/bash
trap 'echo Cancelled; exit' INT
model=$1

for file in inputs/*.txt; do
    python3 predict_tts.py \
        -f $file \
        --outdir outputs/$model \
        --config config/$model/session_paths.yaml
done
