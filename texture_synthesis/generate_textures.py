import torch
from define_algorithms import noise_images, victor_textures, portilla_textures, gatys_textures
import os
import logging
import pickle
from PIL import Image
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATYS_PATH = ROOT / "gatys_model"
sys.path.insert(0, str(GATYS_PATH))

from dataloader_dtd import loader
from dataloader_gaussian import gaussian_loader
from vgg19_arch import VGG19_representations

logging.disable(logging.CRITICAL)

#input paths
gray_images = "/your/gray_images/path"
gray_dir = sorted(os.listdir(gray_images))

bool_images = "/your/binary_images/path"
bool_dir = sorted(os.listdir(bool_images))

#params
parameters = ["gamma",
                "beta1", 
                "beta2",
                "beta3",
                "beta4",
                "theta1",
                "theta2",
                "alpha"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG19_representations().to(device)

def denormalize(input, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    denorm = input*std+mean

    return denorm 

noise_images()


victor_textures()

for fname in gray_dir:
    img_path = os.path.join(gray_images, fname)
    
    portilla_synth = portilla_textures(
        image_path=img_path,
    )

gatys_synth = gatys_textures(
    denorm_function=denormalize,
    model=model,
    data_loader=loader,
    gaussian_loader=gaussian_loader,
)

