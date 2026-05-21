import os
import torch
import matplotlib.pyplot as plt
from clamp import clip_hidden_torch
import matplotlib as mpl
import numpy as np

#params
device = "cuda" if torch.cuda.is_available() else "cpu"

#input path
features_root = "/your/features/path"

#output paths
plot_path = "/your/plots/path"
os.makedirs(plot_path, exist_ok=True)

models_roots = [
    os.path.join(features_root, "vgg19"),
    os.path.join(features_root, "clip"),
    os.path.join(features_root, "dino"),
    os.path.join(features_root, "igpt")
]

def feature_preprocess(
        plot=True,
        plot_path=plot_path,
):
    for mr in models_roots:
        model_name = os.path.basename(mr)
        print(model_name)

        datasets = sorted(
            d for d in os.listdir(mr)
            if os.path.isdir(os.path.join(mr, d))
        )

        for dataset_name in datasets:

            ds_dir = os.path.join(mr, dataset_name)
            print(ds_dir)

            if model_name == "vgg19":
                continue

            elif model_name == "clip" or model_name == "dino" or model_name == "igpt":
                fname = f"{model_name}_features_{dataset_name}.pt"
                features_path = os.path.join(ds_dir, fname)
                features = torch.load(features_path, map_location="cpu")
                print(f"orig features shape {features.shape}")

                #clamp
                vecs = features.cpu().numpy() #[B, L, H]

                del features

                vecs_clamp = clip_hidden_torch(vecs) #hidden_dim size
                clamp_path = os.path.join(ds_dir, f"clamp_{model_name}_features_{dataset_name}.pt")
                torch.save(torch.from_numpy(vecs_clamp), clamp_path)
                print(f"clamped features shape {vecs_clamp.shape}")

                for fname in os.listdir(ds_dir):
                    if (
                        fname.startswith(f"{model_name}_features")
                        and fname.endswith(".pt")
                        and fname != f"vgg19_features_{dataset_name}.pt"
                        and fname != f"clamp_{model_name}_features_{dataset_name}.pt"
                    ):
                        os.remove(os.path.join(ds_dir, fname))


                #average over batch
                vecs_mean = vecs.mean(axis=0) #[L, H]           
                vecs_clamp_mean = vecs_clamp.mean(axis=0)
                print(f"mean over idx clamped features shape {vecs_clamp_mean.shape}")
                    
                if plot:
                    #orig features
                    plt.figure(figsize=(10, 5))

                    layers = vecs_mean.shape[0] #get L
                    cmap = mpl.colormaps.get_cmap("tab20")
                    colors = cmap(np.linspace(0, 1, layers))

                    for l in range(layers):
                        plt.plot(vecs_mean[l], color=colors[l], alpha=0.7, label=f"layer {l}")

                    plt.title(f"{model_name} {dataset_name} features")
                    plt.xlabel("Hidden dimension")
                    plt.ylabel("Value")
                    plt.legend(fontsize=7, ncol=3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_path, f"{model_name}_{dataset_name}_orig.png"))
                    plt.close()

                    #clamped features
                    plt.figure(figsize=(10, 5))

                    layers_clamp = vecs_clamp_mean.shape[0]
                    colors = cmap(np.linspace(0, 1, layers_clamp))

                    for l in range(layers_clamp):
                        plt.plot(vecs_clamp_mean[l], color=colors[l], alpha=0.7, label=f"layer {l}")

                    plt.title(f"{model_name} {dataset_name} clamp features")
                    plt.xlabel("Hidden dimension")
                    plt.ylabel("Value")
                    plt.legend(fontsize=7, ncol=3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_path, f"clamp_{model_name}_{dataset_name}.png"))
                    plt.close()

            else: 
                raise ValueError("some problem with features")

feature_preprocess()