import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image

def generate_faces(count=4, output_dir="output/generated"):
    """Genereaza fete anime folosind StyleGAN2 preantrenat"""
    import torch
    import pickle

    os.makedirs(output_dir, exist_ok=True)
    model_path = "/workspace/ai-upscaling/anime_gan/stylegan2_anime.pkl"

    if not os.path.exists(model_path):
        print("Modelul GAN nu e descarcat! Ruleaza generate_anime.sh mai intai.")
        sys.exit(1)

    print(f"=== Generare {count} fete anime ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Folosim: {device}")

    with open(model_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)

    paths = []
    for i in range(count):
        z = torch.randn(1, G.z_dim).to(device)
        label = torch.zeros([1, G.c_dim]).to(device)

        with torch.no_grad():
            img = G(z, label, truncation_psi=0.7)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()
        path = os.path.join(output_dir, f"face_{i+1:02d}.png")
        Image.fromarray(img_np, "RGB").save(path)
        print(f"  Salvat: {path}")
        paths.append(path)

    return paths

def upscale_to_512(input_paths, output_dir="output/upscaled"):
    """Upscale fete generate la 512x512 cu Real-ESRGAN anime"""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    os.makedirs(output_dir, exist_ok=True)
    model_path = "/workspace/ai-upscaling/weights/RealESRGAN_x4plus_anime_6B.pth"

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=True
    )

    print("=== Upscaling la 512x512 ===")
    for path in input_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        output, _ = upsampler.enhance(img, outscale=2)

        # Resize exact la 512x512
        output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, output)
        print(f"  Salvat: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=4, help="Cate fete sa genereze")
    args = parser.parse_args()

    generated = generate_faces(count=args.count)
    upscale_to_512(generated)
    print("\n=== Pipeline complet! Pozele sunt in output/upscaled/ ===")
