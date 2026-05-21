import os
#comment the following line if running on GPU
os.environ["JAX_PLATFORMS"] = "cpu"

import os
import sys
import torch
import numpy as np
import jax.numpy as jnp

this_dir = os.path.dirname(__file__)
geom_dir = os.path.join(this_dir, "syn-sem")
sys.path.append(geom_dir)

from geometry import pairwise_normalized_l2

models = {
    "vgg19": "VGG-19",
    "clip": "CLIP",
    "dino": "DINO-v2",
    "igpt": "iGPT",
}

target_models = {
    "vgg19": "VGG-19",
    "clip": "CLIP",
    "dino": "DINO-v2",
    "igpt": "iGPT",
}

datasets = {
    "noise_images": "Noise",
    "victor_textures": "V&C",
    "portilla_textures": "P&S",
    "gatys_textures": "G",
    "dtd": "DTD",
    "imagenet_val": "Object",
}

target_datasets = {
    "noise_images": "Noise",
    "victor_textures": "Victor&Conte",
    "portilla_textures": "Portilla&Simoncelli",
    "gatys_textures": "Gatys",
    "dtd": "DTD",
    "imagenet_val": "Object",
}

def get_vgg_similarity(
    ds_dir,
    model_name,
    dataset_name,
    subsample,
    rows, 
    cols,
    distance=pairwise_normalized_l2,
):
    
    if model_name == "vgg19":
        
        n_layers = [1, 4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41, 44, 47, 50]
        L = len(n_layers)
        print(L) 

        subsample = np.asarray(subsample, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        #global indices in the dataset (0-5639)
        g_rows = subsample[rows]
        g_cols = subsample[cols]

        sim_by_layer = []
        
        for i, l in enumerate(sorted(n_layers)):
            fname = f"{model_name}_features_{dataset_name}_layer_{l}.pt"
            features_path = os.path.join(ds_dir, fname)

            print(f"Loading {features_path}")

            try:
                feature = torch.load(features_path, map_location="cpu")
                print(feature.shape)
            except Exception as e:
                raise RuntimeError(f"Failed to load feature file: {features_path}") from e        
                
            A = feature[g_rows].reshape(len(g_rows), -1).to(torch.float32).numpy()
            B = feature[g_cols].reshape(len(g_cols), -1).to(torch.float32).numpy()

            del feature

            sim = distance(jnp.array(A), jnp.array(B))
            sim_by_layer.append(sim)

    return sim_by_layer

#function for mapping relative depths of all models. 0-1 10 bins
def relative_depth(
        layer,
        bins=10,
):       
    depth = np.linspace(0.0, 1.0, bins)
    depth_idx = np.rint(depth*(layer-1)).astype(int)

    return depth, depth_idx, bins