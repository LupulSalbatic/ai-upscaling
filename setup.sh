#!/bin/bash
# Setup Real-ESRGAN pe Vast.ai

echo "=== Instalare dependinte ==="
pip install torch torchvision basicsr facexlib gfpgan
pip install realesrgan

echo "=== Descarcare modele ==="
mkdir -p weights
wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/
wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights/

echo "=== Gata! ==="
