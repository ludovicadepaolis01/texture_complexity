import os 
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader, Subset 
from torchvision.transforms import Compose 
from PIL import Image 

subset_size = 10
batch_size = 8 

root = "your/path/to/images" 
texture_dir = os.listdir(root) 

#prepare images 
img_paths = []
img_dict = {}
for texture_class in texture_dir:
    #collect image paths
    texture_path = os.path.join(root, texture_class)
    img_names = os.listdir(texture_path)
    for fname in img_names:
        img_path = os.path.join(texture_path, fname)
        try:
            with Image.open(img_path) as image:
                img_dict[img_path] = image.copy()
                img_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}") 
        
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
        if isinstance(img, Image.Image): 
            return self.transform(img), path
        else:
            raise ValueError("Expected a PIL.Image object.") 
        
#define subsets in case needed for toy model 
subset_size = subset_size 

dataset = ImgDataset(img_dict, resize=resize) 
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 

subset_indices = list(range(len(dataset)))[:subset_size] 
subset = Subset(dataset, subset_indices) 
subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True) 