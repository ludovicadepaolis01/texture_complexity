import os
import torch
from transformers import CLIPModel, AutoModel, ImageGPTForCausalImageModeling
from transformers import CLIPProcessor, AutoImageProcessor, ImageGPTImageProcessor
import torch.nn.functional as F
from data_preprocess import vits_data_preprocess
from pathlib import Path
import sys

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

imagenet_path = "/your/Object/path" 
imagenet_dir = os.listdir(imagenet_path)
imagenet_basename = os.path.basename(imagenet_path)
print(imagenet_basename)

portilla_path = "/your/P&S/path"
portilla_dir = os.listdir(portilla_path)
portilla_basename = os.path.basename(portilla_path)
print(portilla_basename)

gatys_path = "/your/G/path"
gatys_dir = os.listdir(gatys_path)
gatys_basename = os.path.basename(gatys_path)
print(gatys_basename)

noise_path = "/your/Noise/path" 
noise_dir = os.listdir(noise_path)
noise_basename = os.path.basename(noise_path)
print(noise_basename)

#clip dirs
clip_dir = "/your/CLIP/path"
clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_dir)
print("loaded clip")

#dino dirs
dino_dir = "/your/DINO-v2/path"
dino_model = AutoModel.from_pretrained(dino_dir).to(device)
dino_processor = AutoImageProcessor.from_pretrained(dino_dir)
print("loaded dino")

#igpt dirs
igpt_dir = "/your/iGPT/path"
igpt_model = ImageGPTForCausalImageModeling.from_pretrained(igpt_dir).to(device)
igpt_processor = ImageGPTImageProcessor.from_pretrained(igpt_dir)
print("loaded igpt")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#vits 
def vits_feature_extraction(data_path,
                            data_dir,
                            basename,
                            model_name,
                            model,
                            processor,
                            vits_data_preprocess=vits_data_preprocess,
                            device=device):
        
    model_features_path = f"/your/features/path/{model_name}/{basename}"
    if not os.path.exists(model_features_path):
        os.makedirs(model_features_path)

    model = model.to(device)
    model.eval()

    #get preprocessed data
    image_loader, _ = vits_data_preprocess(data_path,
                                        data_dir,
                                        processor,
                                        model_name)

    #vit feature extraction
    image_features_list = []
    with torch.no_grad():
        for batch in image_loader:
            batch = batch.to(device)

            if model_name == "clip":
                outputs = model.vision_model(pixel_values=batch, output_hidden_states=True)
                feats = torch.stack([output[:, 1:, :].mean(dim=1) for output in outputs.hidden_states], dim=1)

            elif model_name == "dino":
                outputs = model(pixel_values=batch, output_hidden_states=True)
                feats = torch.stack([output[:, 1:, :].mean(dim=1) for output in outputs.hidden_states], dim=1)
                
            elif model_name == "igpt":
                outputs = model(input_ids=batch, output_hidden_states=True, return_dict=True)
                feats = torch.stack([output.mean(dim=1) for output in outputs.hidden_states], dim=1)

            else:
                raise ValueError("wrong model")

            image_features_list.append(feats.cpu())

    image_features = torch.cat(image_features_list, dim=0)
    print(f"{model_name} image features extracted:", image_features.shape)
    save_path = os.path.join(model_features_path, f"{model_name}_features_{basename}.pt")
    torch.save(image_features, save_path, _use_new_zipfile_serialization=False)

paths = [dtd_path, gatys_path, portilla_path, victor_path, noise_path, imagenet_path]
dirs  = [dtd_dir, gatys_dir, portilla_dir, victor_dir, noise_dir, imagenet_dir]
bases = [dtd_basename, gatys_basename, portilla_basename, victor_basename, noise_basename, imagenet_basename]

vit_models = [clip_model, dino_model, igpt_model]
vit_processors = [clip_processor, dino_processor, igpt_processor]
vit_names = ["clip", "dino", "igpt"]

for data_path, data_dir, basename in zip(paths, dirs, bases):
    
    for vm, vp, vn in zip(vit_models, vit_processors, vit_names):
        vits_feature_extraction(
            data_path=data_path,
            data_dir=data_dir,
            basename=basename,
            model_name=vn,
            model=vm,
            processor=vp,
        )
        vm = vm.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
