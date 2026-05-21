import os
#comment the following line if running on GPU
os.environ["JAX_PLATFORMS"] = "cpu"

import os
import sys
import torch
import numpy as np
from jax import random
import jax.numpy as jnp
import pandas as pd
from ii_helpers import get_similarities, plot_grid, get_vgg_similarity, relative_depth

this_dir = os.path.dirname(__file__)
geom_dir = os.path.join(this_dir, "syn-sem")
sys.path.append(geom_dir)

from geometry import (pairwise_normalized_l2,
                      build_information_imbalance,
                      mapped_compute_ranks,
                      )

key = random.PRNGKey(42)
sample_size = 5640 
k = 1

#input paths
features_root = "/your/features/path"

#output paths
plot_path = "/your/plots/path"
csv_path = "/your/csvs/path"

for d in [plot_path, csv_path]:
    os.makedirs(d, exist_ok=True)

models_roots = {
    "vgg19": os.path.join(features_root, "vgg19"),
    "clip": os.path.join(features_root, "clip"),
    "dino": os.path.join(features_root, "dino"),
    "igpt": os.path.join(features_root, "igpt"),
}   

#=========================
#WITHIN MODEL II ANALYSIS
#=========================
def within_model_ii(
        key=key,
        sample_size=sample_size,
        get_vgg_similarity=get_vgg_similarity,
        plot_grid=plot_grid,
        get_similarities=get_similarities,
        csv_path=csv_path,
):
    
    key, subkey = random.split(key)
    subsample = np.array(
        random.choice(subkey, 5640, shape=(sample_size,), replace=False),
    )

    n_row = sample_size // 2
    n_col = sample_size - n_row

    key, subkey = random.split(key)
    rows = np.array(
        random.choice(subkey, sample_size, shape=(n_row,), replace=False),
    )

    remaining = np.setdiff1d(np.arange(sample_size,), rows)

    key, subkey = random.split(key)
    cols = np.array(
        random.choice(subkey, a=remaining, shape=(n_col,), replace=False),
    )

    for model_name, model_path in models_roots.items():
        print(model_name)
        datasets = sorted(d for d in os.listdir(model_path)
                          if os.path.isdir(os.path.join(model_path, d)))
        
        for dataset_name in datasets:
            
            if dataset_name in ("dtd", 
                                "gatys_textures", 
                                "portilla_textures",
                                "victor_textures",
                                "noise_images"):
                continue

            print(dataset_name)
            ds_dir = os.path.join(model_path, dataset_name)

            get_ranks = mapped_compute_ranks(method="min")
            get_ii = build_information_imbalance(k=k)

            if model_name in ("vgg19"):
                 
                similarities = get_vgg_similarity(
                    ds_dir=ds_dir,
                    model_name="vgg19",
                    dataset_name=dataset_name,
                    subsample=subsample,
                    rows=rows,
                    cols=cols,
                )

                L = len(similarities)

                II = np.zeros((L, L))
                IIstd = np.zeros((L, L))
                II_XY = np.zeros((L, L))
                II_YX = np.zeros((L, L))

                rows_out = []
                for l1 in range(L):
                    sim_X = similarities[l1] 
                    for l2 in range(L):
                        sim_Y = similarities[l2]

                        ranks_X, ranks_Y = get_ranks(sim_X, sim_Y)
                        ii, std = get_ii(ranks_X, ranks_Y)

                        print(f"layer {l1} layer {l2}, ii: {ii}")   

                        ii_np  = np.array(ii)
                        std_np = np.array(std)

                        II[l1, l2] = float(np.mean(ii_np))
                        IIstd[l1, l2] = float(np.mean(std_np))
                        II_XY[l1, l2] = float(ii_np[0])
                        II_YX[l1, l2] = float(ii_np[1])

                        rows_out.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "layer_1": l1,
                            "layer_2": l2,
                            "II_mean": II[l1, l2],
                            "II_std": IIstd[l1, l2],
                            "II_XY": II_XY[l1, l2],
                            "II_YX": II_YX[l1, l2],
                        })

                data_within_ii = pd.DataFrame(rows_out)
                out_dir = os.path.join(csv_path, f"csv_within_ii")
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"within_ii_{model_name}_{dataset_name}.csv")
                data_within_ii.to_csv(out_file, index=False)
                print(f"saved {out_file}")
            
            elif model_name in ("clip", "dino", "igpt"):

                fname = f"clamp_{model_name}_features_{dataset_name}.pt"
                features_path = os.path.join(ds_dir, fname)
                features = torch.load(features_path, map_location="cpu")
                print(features.shape)
                features = jnp.array(features.numpy())

                _, L, _ = features.shape

                Aidx = subsample[rows] 
                Bidx = subsample[cols]

                similarities = []
                for l in range(L):
                    X_A = features[Aidx, l, :]
                    X_B = features[Bidx, l, :]
                    sim = pairwise_normalized_l2(X_A, X_B)
                    similarities.append(sim)

                II = np.zeros((L, L))
                IIstd = np.zeros((L, L))
                II_XY = np.zeros((L, L))
                II_YX = np.zeros((L, L))

                rows_out = []
                for l1 in range(L):
                    sim_X = similarities[l1]
                    for l2 in range(L):
                        sim_Y = similarities[l2]

                        ranks_X, ranks_Y = get_ranks(sim_X, sim_Y)
                        ii, std = get_ii(ranks_X, ranks_Y)

                        print(f"layer {l1} layer {l2}, ii: {ii}")   

                        ii_np = np.array(ii)
                        std_np = np.array(std)

                        II[l1, l2] = float(ii_np.mean())
                        IIstd[l1, l2] = float(std_np.mean())
                        II_XY[l1, l2] = float(ii_np[0])
                        II_YX[l1, l2] = float(ii_np[1])

                        rows_out.append({
                                    "model": model_name,
                                    "dataset": dataset_name,
                                    "layer_1": l1,
                                    "layer_2": l2,
                                    "II_mean": II[l1, l2],
                                    "II_std": IIstd[l1, l2],
                                    "II_XY": II_XY[l1, l2],
                                    "II_YX": II_YX[l1, l2],
                                })

                        data_within_ii = pd.DataFrame(rows_out)
                        out_dir = os.path.join(csv_path, f"csv_within_ii")
                        os.makedirs(out_dir, exist_ok=True)
                        out_file = os.path.join(out_dir, f"within_ii_{model_name}_{dataset_name}.csv")
                        data_within_ii.to_csv(out_file, index=False)

    print("within model II done")

within_model_ii()


#==========================
#ACROSS MODELS II ANALYSIS
#==========================
def across_models_ii(
    reference_model,
    key=key,
    relative_depth=relative_depth,
    get_similarities=get_similarities,
    csv_path=csv_path,
    bins=10,
):

    get_ranks = mapped_compute_ranks(method="min")
    get_ii = build_information_imbalance(k=k)

    os.makedirs(plot_path, exist_ok=True)

    key, subkey = random.split(key)
    subsample = np.array(
        random.choice(subkey, 5640, shape=(sample_size,), replace=False),
    )

    n_row = sample_size // 2
    n_col = sample_size - n_row

    key, subkey = random.split(key)
    rows = np.array(
        random.choice(subkey, sample_size, shape=(n_row,), replace=False),
    )

    remaining = np.setdiff1d(np.arange(sample_size,), rows)

    key, subkey = random.split(key)
    cols = np.array(
        random.choice(subkey, a=remaining, shape=(n_col,), replace=False),
    )

    Aidx = subsample[rows]
    Bidx = subsample[cols]

    for model_name, mr in models_roots.items():
        print(model_name)

        all_datasets = sorted(
            d for d in os.listdir(mr)
            if os.path.isdir(os.path.join(mr, d))
        )

    for dataset_name in all_datasets:
        print(dataset_name)

        similarities_per_model = {}

        for model_name, mr in models_roots.items():
            ds_dir = os.path.join(mr, dataset_name)

            if model_name == "vgg19":

                sims = get_vgg_similarity(
                    ds_dir=ds_dir,
                    model_name="vgg19",
                    dataset_name=dataset_name,
                    subsample=subsample,
                    rows=rows,
                    cols=cols,
                )

            elif model_name in ("clip", "dino", "igpt"):
                
                fname = f"clamp_{model_name}_features_{dataset_name}.pt"
                features_path = os.path.join(ds_dir, fname)
                features = torch.load(features_path, map_location="cpu")
                features = jnp.array(features.numpy())

                _, L, _ = features.shape

                sims = []
                for l in range(L):
                    X_A = features[Aidx, l, :]
                    X_B = features[Bidx, l, :]
                    sim = pairwise_normalized_l2(X_A, X_B)
                    sims.append(sim)

            similarities_per_model[model_name] = sims

        if reference_model not in similarities_per_model:
            continue

        model_names = list(similarities_per_model.keys())
        X_ref = similarities_per_model[reference_model]
        L_ref = len(X_ref)

        _, idx_ref, _ = relative_depth(L_ref, bins=bins)

        for target_model in model_names:
            print(f"{reference_model} --> {target_model}")
            if target_model == reference_model:
                continue

            Y = similarities_per_model[target_model]
            Lm = len(Y)

            _, idx_ref, _ = relative_depth(L_ref, bins=bins)
            _, idx_m, _ = relative_depth(Lm, bins=bins)

            rows_out = []
            for b in range(bins):
                sim_X = X_ref[idx_ref[b]]
                sim_Y = Y[idx_m[b]]

                ranks_X, ranks_Y = get_ranks(sim_X, sim_Y)
                ii, std = get_ii(ranks_X, ranks_Y)
                print(ii)

                ii_np = np.array(ii)
                std_np = np.array(std)

                rows_out.append({
                    "dataset": dataset_name,
                    "reference_model": reference_model,
                    "target_model": target_model,
                    "bin": b,
                    "ref_layer_idx": int(idx_ref[b]),
                    "target_layer_idx": int(idx_m[b]),
                    "II_mean": float(np.mean(ii_np)),
                    "II_XY": float(ii_np[0]),
                    "II_YX": float(ii_np[1]),
                    "II_std": float(np.mean(std_np)),
                })

            data_across_models_ii = pd.DataFrame(rows_out)
            out_dir = os.path.join(csv_path, f"csv_across_models_ii")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"across_models_ii_{reference_model}_vs_{target_model}_{dataset_name}.csv")
            data_across_models_ii.to_csv(out_file, index=False)
            print(f"saved {out_file}")

    print("across model II done")


referece_models = ["vgg19", "clip", "dino", "igpt"]

for ref_model in referece_models:
    across_models_ii(ref_model)


#============================
#ACROSS DATASETS II ANALYSIS
#============================
def across_data_ii(
    key=key,
    relative_depth=relative_depth,
    get_similarities=get_similarities,
    csv_path=csv_path,
    bins=10,
):
        
    get_ranks = mapped_compute_ranks(method="min")
    get_ii = build_information_imbalance(k=k)

    os.makedirs(plot_path, exist_ok=True)

    key, subkey = random.split(key)
    subsample = np.array(
        random.choice(subkey, 5640, shape=(sample_size,), replace=False),
    )

    n_row = sample_size // 2
    n_col = sample_size - n_row

    key, subkey = random.split(key)
    rows = np.array(
        random.choice(subkey, sample_size, shape=(n_row,), replace=False),
    )

    remaining = np.setdiff1d(np.arange(sample_size,), rows)

    key, subkey = random.split(key)
    cols = np.array(
        random.choice(subkey, a=remaining, shape=(n_col,), replace=False),
    )

    Aidx = subsample[rows]
    Bidx = subsample[cols]

    for model_name, mr in models_roots.items():
        print(model_name)

        all_datasets = sorted(
            d for d in os.listdir(mr)
            if os.path.isdir(os.path.join(mr, d))
        )

        similarities_per_dataset = {}
        for dataset_name in all_datasets:
            print(dataset_name)

            if dataset_name == "imagenet_val":
                continue

            ds_dir = os.path.join(mr, dataset_name)
            print(ds_dir)

            if not os.path.isdir(ds_dir):
                continue

            if model_name == "vgg19":

                sims = get_vgg_similarity(
                    ds_dir=ds_dir,
                    model_name="vgg19",
                    dataset_name=dataset_name,
                    subsample=subsample,
                    rows=rows,
                    cols=cols,
                )

                similarities_per_dataset[dataset_name] = sims

            elif model_name in ("clip", "dino", "igpt"):

                features_path = os.path.join(ds_dir, f"clamp_{model_name}_features_{dataset_name}.pt")
                features = torch.load(features_path, map_location="cpu")
                print(features.shape)
                features = jnp.array(features.numpy())

                _, L, _ = features.shape

                sims = []
                for l in range(L):
                    X_A = features[Aidx, l, :]
                    X_B = features[Bidx, l, :]
                    sim = pairwise_normalized_l2(X_A, X_B)  # (n_row, n_col)
                    sims.append(sim)

                similarities_per_dataset[dataset_name] = sims
            
        for ref_dataset in all_datasets:

            if ref_dataset == "imagenet_val":
                continue

            if ref_dataset not in similarities_per_dataset:
                continue

            print(ref_dataset)

            X_ref = similarities_per_dataset[ref_dataset]
            L_ref = len(X_ref)

            _, idx_ref, _ = relative_depth(L_ref, bins=bins)

            for dataset in all_datasets:
                print(f"{ref_dataset} --> {dataset}")
                if dataset == ref_dataset:
                    continue

                if dataset not in similarities_per_dataset:
                    continue

                Y = similarities_per_dataset[dataset]
                Lm = len(Y)
  
                _, idx_ref, _ = relative_depth(L_ref, bins=bins)
                _, idx_m, _ = relative_depth(Lm, bins=bins)

                rows_out = []

                for b in range(bins):
                
                    sim_X = X_ref[idx_ref[b]]
                    sim_Y = Y[idx_m[b]]

                    ranks_X, ranks_Y = get_ranks(sim_X, sim_Y)
                    ii, std = get_ii(ranks_X, ranks_Y)

                    print(ii)

                    ii_np = np.array(ii)
                    std_np = np.array(std)

                    rows_out.append({
                        "model": model_name,
                        "ref_dataset": ref_dataset,
                        "target_dataset": dataset,
                        "bin": b,
                        "ref_layer_idx": int(idx_ref[b]),
                        "target_layer_idx": int(idx_m[b]),
                        "II_mean": float(np.mean(ii_np)),
                        "II_XY": float(ii_np[0]),
                        "II_YX": float(ii_np[1]),
                        "II_std": float(np.mean(std_np)),
                    })

                data_across_data_ii = pd.DataFrame(rows_out)
                out_dir = os.path.join(csv_path, f"csv_across_data_ii")
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"across_data_ii_{model_name}_{ref_dataset}_vs_{dataset}.csv")
                data_across_data_ii.to_csv(out_file, index=False)
                print(f"saved {out_file}")

    print("across data II done")

across_data_ii()