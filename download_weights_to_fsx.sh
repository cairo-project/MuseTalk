#!/bin/bash
set -e

CheckpointsDir="/fsx/shared/users/landz/models/Musetalk"

mkdir -p \
  "$CheckpointsDir/musetalk" \
  "$CheckpointsDir/musetalkV15" \
  "$CheckpointsDir/syncnet" \
  "$CheckpointsDir/dwpose" \
  "$CheckpointsDir/face-parse-bisent" \
  "$CheckpointsDir/sd-vae" \
  "$CheckpointsDir/whisper"

# MuseTalk V1.0 weights
hf download TMElyralab/MuseTalk musetalk/musetalk.json musetalk/pytorch_model.bin \
  --local-dir "$CheckpointsDir"

# MuseTalk V1.5 weights
hf download TMElyralab/MuseTalk musetalkV15/musetalk.json musetalkV15/unet.pth \
  --local-dir "$CheckpointsDir"


# SD VAE weights
hf download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin \
  --local-dir "$CheckpointsDir/sd-vae"

# Whisper weights
hf download openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json \
  --local-dir "$CheckpointsDir/whisper"

# DWPose weights
hf download yzd-v/DWPose dw-ll_ucoco_384.pth \
  --local-dir "$CheckpointsDir/dwpose"

# SyncNet weights
hf download ByteDance/LatentSync latentsync_syncnet.pt \
  --local-dir "$CheckpointsDir/syncnet"

# Face Parse BiSeNet weights
pip install -q gdown
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$CheckpointsDir/face-parse-bisent/79999_iter.pth"
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth"

echo "All weights downloaded to $CheckpointsDir"
