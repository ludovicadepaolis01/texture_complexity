import metex
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import shutil
import sys
import gc
import torch
import torchvision.utils as u
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from binary_data_preprocess import texture_coords, histogram

portilla_script = str(
    Path("your/path/to/texture_models/portilla_model/")
    / "texture_synthesis_g.py"
)

ROOT = Path(__file__).resolve().parents[1]
GATYS_PATH = ROOT / "gatys_model"
sys.path.insert(0, str(GATYS_PATH))

#input paths
bool_path = "/your/binary_images/path"

#output paths
noise_images_out = "/your/noise_images/path"
victor_images_out = "/your/victor_textures/path"
portilla_images_out = "/your/portilla_textures/path"
gatys_images_out = "/your/gatys_textures/path"

#checkpoint paths
gatys_checkpoint_path = "your/checkpoints/path"
Path(gatys_checkpoint_path).mkdir(parents=True, exist_ok=True)

for d in [noise_images_out, victor_images_out, portilla_images_out, gatys_images_out]:
    os.makedirs(d, exist_ok=True)

#params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==============================
#DEFINE NOISE IMAGES SYNTHESIS
#==============================
def noise_images(out_path=noise_images_out,
                bool_path=bool_path, 
                height=256):
    
    os.makedirs(out_path, exist_ok=True)

    for fname in sorted(os.listdir(bool_path)):
        if not fname.endswith(".npy"):
            continue

        base = fname.replace(".npy", "")          
        texture_name = base.split("_")[0]         

        texture_folder = os.path.join(out_path, texture_name)
        os.makedirs(texture_folder, exist_ok=True)

        tex = metex.Texture(height=height, gamma=0).sample()
        tex = np.array(tex) * 255

        out_fname = f"noise_{base}.png"
        plt.imsave(os.path.join(texture_folder, out_fname), tex, cmap="gray")

#======================
#DEFINE V&C ALGORITHM
#======================
def victor_textures(out_path=victor_images_out,
                    texture_coords=texture_coords, 
                    histogram=histogram,
                    bool_path=bool_path, 
                    height=256):
    
    os.makedirs(out_path, exist_ok=True)

    coord_names = ["gamma","beta1","beta2","beta3","beta4",
                   "theta1","theta2","theta3","theta4","alpha"]

    for fname in sorted(os.listdir(bool_path)):
        if not fname.endswith(".npy"):
            continue

        X = np.load(os.path.join(bool_path, fname)).astype(np.bool_)
        c = texture_coords(histogram(X))

        i = int(np.argmax(np.abs(c)))
        pname = coord_names[i]
        pval = float(c[i])

        tex = metex.Texture(height=height, **{pname: pval}).sample()
        tex = np.array(tex) * 255  # 0/255 for saving

        base = fname.replace(".npy", "")          
        texture_name = base.split("_")[0]         

        texture_folder = os.path.join(out_path, texture_name)
        os.makedirs(texture_folder, exist_ok=True)

        out_fname = f"v_{base}.png"
        plt.imsave(os.path.join(texture_folder, out_fname), tex, cmap="gray")

#=====================
#DEFINE P&S ALGORITHM
#=====================
def portilla_textures(
        image_path,
):
    cmd = [
        "python", portilla_script,
        "-i", image_path,
        "-o", portilla_images_out,
        "-n", "5",
        "-k", "4",
        "-m", "7",
        "--iter", "10",
    ]

    subprocess.run(cmd, check=True)

    pattern = os.path.join(portilla_images_out, "out-n5-k4-m7-*.png")
    targets = glob.glob(pattern)
    if not targets:
        raise RuntimeError(f"No Portilla output found")

    latest = max(targets, key=os.path.getmtime) #take last generated image
    
    basename = os.path.basename(image_path)
    base = basename.replace("_gray.png", "") #remove "gray" string
    texture_name = base.split("_")[0]

    texture_folder = out_full_path = os.path.join(portilla_images_out, texture_name)
    os.makedirs(texture_folder, exist_ok=True)

    out_fname = f"p_{base}.png"
    out_full_path = os.path.join(texture_folder, out_fname)
    
    shutil.move(latest, out_full_path)
    
    return out_full_path

#===================
#DEFINE G ALGORITHM
#===================
def gatys_textures(denorm_function,
                model,
                data_loader,
                gaussian_loader,
                output_path=gatys_images_out,
                ):

    MSE = torch.nn.MSELoss()
    optim_steps = 30000
    
    model.zero_grad()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    #parameters for images
    mean = (0.5137, 0.4639, 0.4261)
    std = (0.2576, 0.2330, 0.2412)

    for batch_idx, ((orig_batch, path), reco_batch)  in enumerate(zip(data_loader, gaussian_loader)):        
        orig_batch = orig_batch.to(device)
        #define gradient with respect to image as a parameter
        reco_batch = nn.Parameter(reco_batch.clone().detach().to(device)) 
        optimizer = optim.Adam([reco_batch], lr=1e-4)

        first_path = path[0]
        texture_class = os.path.basename(os.path.dirname(first_path))

        #load checkpoint if present (per class/batch)
        ckpt_name = f"{texture_class}_batch{batch_idx}.pt"
        class_checkpoint_path = os.path.join(gatys_checkpoint_path, ckpt_name)
        
        start_step = 0

        if os.path.exists(class_checkpoint_path):
            checkpoint = torch.load(class_checkpoint_path, map_location=device)
            start_step = int(checkpoint.get("step", 0))

            #if this batch is already finished, skip it entirely
            if start_step >= optim_steps:
                continue

            #restore reco image
            if checkpoint.get("reco_image") is not None:
                with torch.no_grad():
                    reco_batch.data.copy_(checkpoint["reco_image"].to(device))

            #restore optimizer
            if checkpoint.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print(f"[RESUME] class={texture_class} batch={batch_idx} from step={start_step}")
        else:
            print(f"[NEW] class={texture_class} batch={batch_idx} (no checkpoint)")
            
        with torch.no_grad():
            orig_gram_matrices, _ = model(orig_batch)

        last_loss = None

        #optimization
        for step in range(start_step, optim_steps): 
            optimizer.zero_grad()
            reco_gram_matrices, feature_map_m_list = model(reco_batch)

            sum_gram_matrix_loss = 0
            for orig_gram_matrix, reco_gram_matrix, m in zip(orig_gram_matrices, reco_gram_matrices, feature_map_m_list):
                gram_matrix_loss = MSE(orig_gram_matrix, reco_gram_matrix)/(4*m)
                sum_gram_matrix_loss += gram_matrix_loss

            #backward pass over the images and update optimizer
            sum_gram_matrix_loss.backward()
            optimizer.step()
            last_loss = sum_gram_matrix_loss.item()

            #save checkpoint regularly and at the very end
            if (step + 1) % 10 == 0 or (step + 1) == optim_steps:
                torch.save({
                    "step": step + 1,
                    "reco_image": reco_batch.detach().cpu(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, class_checkpoint_path)

        with torch.no_grad():
            #denormalize orig an reco images
            denorm_image = denorm_function(orig_batch, mean, std).to(device)
            denorm_reco = denorm_function(reco_batch, mean, std).to(device)

        for idx, (one_orig, one_path, one_reco) in enumerate(zip(denorm_image, path, denorm_reco)):
            basename = os.path.basename(one_path)
            base = basename.replace(".jpg", "")
            print(base)

            texture_folder = output_path = os.path.join(gatys_images_out, texture_class)
            os.makedirs(texture_folder, exist_ok=True)

            out_fname = f"g_{base}.png"
            out_fname_orig = f"g_{base}_orig.png"

            output_path = os.path.join(texture_folder, out_fname)
            output_path_orig = os.path.join(texture_folder, out_fname_orig)

            reco = u.save_image(one_reco, output_path)
            orig = u.save_image(one_orig, output_path_orig)

        print(f"optimization_steps: {optim_steps}, texture: {texture_class}, gram loss: {last_loss}")
        torch.cuda.empty_cache()
        gc.collect()
