import os
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from pathlib import Path
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

#gatys model dir
ROOT = Path(__file__).resolve().parents[1]
GATYS_PATH = ROOT / "gatys_model"
sys.path.insert(0, str(GATYS_PATH))

from dataloader_dtd import loader
from dataloader_gaussian import gaussian_loader
from vgg19_arch import VGG19_representations

#input paths
dtd_path = "/your/DTD/path"
dtd_dir = os.listdir(dtd_path)

victor_path = "/your/V&C/path"
victor_dir = os.listdir(victor_path)

portilla_path = "/your/P&S/path"
portilla_dir = os.listdir(portilla_path)

gatys_path = "/your/G/path"
gatys_dir = os.listdir(gatys_path)

imagenet_path = "/your/Object/path" 
imagenet_dir = os.listdir(imagenet_path)

noise_path = "/your/Noise/path"
noise_dir = os.listdir(noise_path)

#vgg19_gatys data preprocess
def vgg_data_preprocess(dataset_path,
                        batch_size=8,
                        subset_size=10,
                        ):
    
    texture_dir = os.listdir(dataset_path)

    img_paths = []
    img_dict = {}
    
    if dataset_path in (dtd_path, gatys_path, portilla_path, victor_path, noise_path):
        for texture_class in texture_dir:
            #collect image paths
            texture_path = os.path.join(dataset_path, texture_class)
            img_names = os.listdir(texture_path)

            for fname in img_names:
                img_path = os.path.join(texture_path, fname)
                try:
                    with Image.open(img_path) as image:
                        img_dict[img_path] = image.copy()
                        img_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}") 
                                                              
    elif dataset_path == imagenet_path:
        img_names = sorted(os.listdir(dataset_path))

        for fname in img_names:
            img_path = os.path.join(dataset_path, fname)
            try:
                with Image.open(img_path) as image:
                    img_dict[img_path] = image.copy()
                    img_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}") 

    else:
        pass
                
    #params for image transformations 
    resize = 224 
    image_index = 0 

    class ImgDataset(Dataset): 
        def __init__(self, img_dict, resize=resize): 
            self.img_paths = list(img_dict.keys())
            self.images = list(img_dict.values())

            #add a preload transformation variable that contains the heaviest transformations  
            self.transform = Compose([
                transforms.Resize((resize, resize)), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5137, 0.4639, 0.4261), 
                                    (0.2576, 0.2330, 0.2412)),
                ])
            
        def __len__(self):
            return len(self.img_paths)
        
        def __getitem__(self, idx): 
            img = self.images[idx]
            path = self.img_paths[idx]

            if img.mode != "RGB":
                img = img.convert("RGB")

            if isinstance(img, Image.Image): 
                return self.transform(img), path
            else:
                raise ValueError("Expected a PIL.Image object.") 

    #define subsets in case needed for toy model 
    subset_size = subset_size 

    dataset = ImgDataset(img_dict, resize=resize) 
    loader = DataLoader(dataset,
                        batch_size=batch_size, 
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True,
                        ) 

    subset_indices = list(range(len(dataset)))[:subset_size] 
    subset = Subset(dataset, subset_indices) 
    subset_loader = DataLoader(subset, 
                               batch_size=batch_size, 
                               shuffle=False,
                               num_workers=1,
                               pin_memory=True,
                               )

    return loader, subset_loader

def vits_data_preprocess(img_path,
                        img_dir,
                        processor,
                        model_name,
                        batch_size=8,
                        ):
    
    img_list = []
    img_names_list = []

    if img_path in (dtd_path, gatys_path, portilla_path, victor_path, noise_path):
        for texture_class in img_dir:
            class_path = os.path.join(img_path, texture_class)
            if os.path.isdir(class_path):
                class_dir = sorted(os.listdir(class_path))
                for img in class_dir:
                    img_names_list.append(img)
                    file_path = os.path.join(class_path, img)
                    try:
                        with Image.open(file_path) as image:
                            img_list.append(image.copy())
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        img_index = len(img_list)

    elif img_path == imagenet_path:
        imagenet_dir = sorted(os.listdir(img_path))
        for img in imagenet_dir:
            img_names_list.append(img)
            file_path = os.path.join(img_path, img)
            try:
                with Image.open(file_path) as image:
                    img_list.append(image.copy())
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        img_index = len(img_list)
    
    else:
        pass

    #prepreocess vits images
    def preprocess(img):

        if img.mode != "RGB":
                img = img.convert("RGB")

        if model_name == "clip" or model_name == "dino":
            images = processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        elif model_name == "igpt":
            images = processor(images=img, return_tensors="pt")["input_ids"].squeeze(0)

        return images

    class ImageDataset(Dataset):
        def __init__(self, img_list, preprocess):
            self.img_list = img_list
            self.preprocess = preprocess

        def __len__(self):
            return len(self.img_list)

        def __getitem__(self, idx):
            image = self.preprocess(self.img_list[idx])
            return image

    #instantiate the dataloaders
    image_dataset = ImageDataset(img_list, preprocess)
    
    idx = 50  #or any integer < len(img_list)
    subset_indices = list(range(idx))
    subset = Subset(image_dataset, subset_indices) 

    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    subset_indices = list(range(min(len(image_dataset), 10)))
    subset = Subset(image_dataset, subset_indices)
    imagenet_subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    return image_loader, imagenet_subset_loader