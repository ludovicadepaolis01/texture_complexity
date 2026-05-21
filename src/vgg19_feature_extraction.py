import os
import torch
import torch.nn.functional as F
from data_preprocess import vgg_data_preprocess
from pathlib import Path
import sys
import argparse

#params
device = "cuda" if torch.cuda.is_available() else "cpu"

#input paths
dtd_path = "/your/DTD/path"
dtd_dir = os.listdir(dtd_path)
dtd_basename = os.path.basename(os.path.dirname(dtd_path))
print(dtd_basename)

victor_path = "/your/V&C/path"
victor_dir = os.listdir(victor_path)
victor_basename = os.path.basename(victor_path)
print(victor_basename)

portilla_path = "/your/P&S/path"
portilla_dir = os.listdir(portilla_path)
portilla_basename = os.path.basename(portilla_path)
print(portilla_basename)

gatys_path = "/your/G/path"
gatys_dir = os.listdir(gatys_path)
gatys_basename = os.path.basename(gatys_path)
print(gatys_basename)

imagenet_path = "/your/Object/path" 
imagenet_dir = os.listdir(imagenet_path)
imagenet_basename = os.path.basename(imagenet_path)
print(imagenet_basename)

noise_path = "/your/Noise/path" 
noise_dir = os.listdir(noise_path)
noise_basename = os.path.basename(noise_path)
print(noise_basename)

#gatys model dir
ROOT = Path(__file__).resolve().parents[1]
GATYS_PATH = ROOT / "gatys_model"
sys.path.insert(0, str(GATYS_PATH))
vgg_name = "vgg19"

from vgg19_features import VGG19_features, layer_indices
print("loaded vgg19")
print(layer_indices)

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, choices=layer_indices, 
                    help="Which vgg19 layer to extract")

args = parser.parse_args()
idx = args.index

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#vgg19
def vgg_features_extraction(data_path,
                            basename,
                            model_name,
                            selected_idx=idx,
                            vgg_data_preprocess=vgg_data_preprocess,
                            vgg_model=VGG19_features,
                            ):
    
    model = vgg_model(selected_idx).to(device)
    model.eval()

    model_features_path = f"/your/features/path/{model_name}/{basename}"
    if not os.path.exists(model_features_path):
        os.makedirs(model_features_path, exist_ok=True)

    image_loader, _ = vgg_data_preprocess(data_path)

    chunks = []
    with torch.no_grad():
        for images, _ in image_loader:
            images = images.to(device, non_blocking=True)
            features = model(images)
            features = features.to(torch.float16).cpu()
            print(features.shape)
            chunks.append(features)

            print(f"batch {features.shape}")
            del images, features

    full_features = torch.cat(chunks, dim=0)
    torch.save(full_features, 
               os.path.join(model_features_path, f"{model_name}_features_{basename}_layer_{selected_idx}.pt"), 
               _use_new_zipfile_serialization=False)
    print(f"{model_name} image features extracted:", full_features.shape)


paths = [dtd_path, gatys_path, portilla_path, victor_path, imagenet_path, noise_path]   
dirs  = [dtd_dir, gatys_dir, portilla_dir, victor_dir, imagenet_dir, noise_dir]
bases = [dtd_basename, gatys_basename, portilla_basename, victor_basename, imagenet_basename, noise_basename]

for data_path, data_dir, basename in zip(paths, dirs, bases):

    vgg_features_extraction(data_path, basename, "vgg19")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
