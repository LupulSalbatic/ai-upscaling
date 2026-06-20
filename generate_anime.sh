#!/bin/bash
# Descarca StyleGAN2 preantrenat pe fete anime
echo "=== Instalare dependinte ==="
pip install ninja

echo "=== Descarcare StyleGAN2-ADA PyTorch ==="
cd /workspace
if [ ! -d "stylegan2-ada-pytorch" ]; then
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
fi

echo "=== Descarcare model anime faces ==="
mkdir -p /workspace/ai-upscaling/anime_gan
cd /workspace/ai-upscaling/anime_gan

# Model StyleGAN2 antrenat pe anime faces (de pe HuggingFace)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
import os

# Incearca sa descarce modelul
try:
    path = hf_hub_download(
        repo_id='hysts/stylegan2-anime',
        filename='model.pkl',
        local_dir='/workspace/ai-upscaling/anime_gan'
    )
    print('Descarcat:', path)
except Exception as e:
    print('Eroare:', e)
    print('Incearca alt repo...')
    try:
        path = hf_hub_download(
            repo_id='ybelkada/anime-stylegan2',
            filename='network-snapshot.pkl',
            local_dir='/workspace/ai-upscaling/anime_gan'
        )
        print('Descarcat:', path)
    except Exception as e2:
        print('Eroare:', e2)
        print('Descarca manual de pe: https://huggingface.co/models?search=anime+stylegan')
"

echo "=== Gata! ==="
