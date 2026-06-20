#!/bin/bash
# Descarca modelul StyleGAN2 preantrenat pe fete anime
echo "=== Descarcare model anime GAN ==="
mkdir -p /workspace/ai-upscaling/anime_gan

pip install gdown

# Model antrenat pe Danbooru anime faces (256x256)
gdown --id 1A4BbJMWBMkAmxOdGoiqbLSAJ_WjGE5PT -O /workspace/ai-upscaling/anime_gan/stylegan2_anime.pkl

echo "=== Gata! ==="
