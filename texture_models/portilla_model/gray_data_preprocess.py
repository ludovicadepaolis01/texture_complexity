import torch
import torchvision.transforms as transforms
import torchvision.utils as u
from PIL import Image
import os
import pickle

#code to prepare gray images for P&S algorithm

resize = 256

#input paths
img_path = "your/dtd/path"
img_dir = os.listdir(img_path)

#output paths
texture_dict_path = "/your/texture_dictionary/path"
save_gray = "/your/gray_images/path"

for d in [texture_dict_path, save_gray]:
    os.makedirs(d, exist_ok=True)

img_dict = {}
for texture_class in img_dir:
    class_path = os.path.join(img_path, texture_class)
    if os.path.isdir(class_path):
        img_list = []
        for img in os.listdir(class_path):
            file_path = os.path.join(class_path, img)
            img_list.append(file_path)
            
            image = Image.open(file_path)
            gray_image = image.convert("L")  
            gray_image = gray_image.resize((resize, resize))
            out_path = os.path.join(save_gray, img.replace(".jpg", "_gray.png"))
            gray_image.save(out_path)
            
        img_dict[texture_class] = img_list

        with open(os.path.join(texture_dict_path, "texture_dict.pkl"), "wb") as f:
                pickle.dump(img_dict, f)  