import argparse
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale(input_path, output_path, scale=4, anime=False):
    if anime:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_path = "weights/RealESRGAN_x4plus_anime_6B.pth"
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = "weights/RealESRGAN_x4plus.pth"

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True
    )

    import cv2
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=scale)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"Salvat: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Poza input")
    parser.add_argument("-o", "--output", default="output/result.png", help="Poza output")
    parser.add_argument("-s", "--scale", type=float, default=4, help="Factor upscale (2, 4, 8)")
    parser.add_argument("--anime", action="store_true", help="Mod anime/desen animat")
    args = parser.parse_args()

    upscale(args.input, args.output, args.scale, args.anime)
