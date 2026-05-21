import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess import vgg_data_preprocess
import matplotlib as mpl
import matplotlib.image as mpimg
import pandas as pd
import csv
import glob

#input paths
dtd_path = "/your/DTD/path"
dtd_dir = os.listdir(dtd_path)
dtd_basename = os.path.basename(os.path.dirname(dtd_path))
print(dtd_basename)

classifier_path = "/your/classifier/path"
if not os.path.exists(classifier_path):
    os.makedirs(classifier_path, exist_ok=True)

features_root = "/your/features/path"

#output paths
plot_path = "/your/plots/path"
csv_path = "/your/csv/path"

for d in [plot_path, csv_path]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

os.makedirs(plot_path, exist_ok=True)
models_roots = [
    os.path.join(features_root, "vgg19"),
    os.path.join(features_root, "clip"),
    os.path.join(features_root, "dino"),
    os.path.join(features_root, "igpt")
]

models = {
    "vgg19": "VGG-19",
    "clip": "CLIP",
    "dino": "DINO-v2",
    "igpt": "iGPT",
}

datasets = {
    "noise_images": "Noise",
    "victor_textures": "Victor&Conte",
    "portilla_textures": "Portilla&Simoncelli",
    "gatys_textures": "Gatys",
    "dtd": "DTD",
    "imagenet_val": "Object",
}

#params
device = "cuda" if torch.cuda.is_available() else "cpu"

#set scientific = False for classification task;
#set scientific = True for transfer learning task
scientific = True

loader_dtd, _ = vgg_data_preprocess(dtd_path, batch_size=8, subset_size=50)

def prepare_data(
        mr,
        dataset_name,  
        loader_dtd=loader_dtd,
        seed=0,
    ):

    #prepare labels
    dataset_dtd = loader_dtd.dataset

    labels_list = []
    for i in range(len(dataset_dtd)):
        _, path = dataset_dtd[i]
        labels_list.append(os.path.basename(os.path.dirname(path)))

    unique_label = []
    for label in labels_list:
        if label not in unique_label:
            unique_label.append(label)

    sorted_ul = sorted(unique_label)
    classes = len(sorted_ul)

    class_to_n = {}
    i = 0
    for cl in sorted_ul:
        class_to_n[cl] = i
        i = i + 1

    y = []
    for texture in labels_list:
        t = class_to_n[texture]
        y.append(t)

    model_name = os.path.basename(mr)
    ds_dir = os.path.join(mr, dataset_name)
    data_by_layer = {}


    if model_name == "vgg19":
        n_layers = [4, 24, 47]

        for l in sorted(n_layers):
            fname = f"{model_name}_features_{dataset_name}_layer_{l}.pt"
            features_path = os.path.join(ds_dir, fname)

            feature = torch.load(features_path, map_location="cpu")

            X = feature.mean(dim=(2, 3))

            dims = X.shape[1]
        
            X_tr, X_te, y_tr, y_te = train_test_split(
                X.numpy(), 
                y, 
                test_size=0.2, 
                random_state=seed, 
                stratify=y)
            
            data_by_layer[l] = (dims, X_tr, X_te, y_tr, y_te)

    elif model_name == "clip":
            
        n_layers = [1, 6, 11]

        for l in sorted(n_layers):
            fname = f"clamp_{model_name}_features_{dataset_name}.pt"
            features_path = os.path.join(ds_dir, fname)
            features = torch.load(features_path, map_location="cpu")[:, l, :]

            dims = features.shape[1]

            X = features

            X_tr, X_te, y_tr, y_te = train_test_split(
                X.numpy(), 
                y, 
                test_size=0.2, 
                random_state=seed, 
                stratify=y)
            
            data_by_layer[l] = (dims, X_tr, X_te, y_tr, y_te)

    elif model_name == "dino":

        n_layers = [2, 12, 22]

        for l in sorted(n_layers):
            fname = f"clamp_{model_name}_features_{dataset_name}.pt"
            features_path = os.path.join(ds_dir, fname)
            features = torch.load(features_path, map_location="cpu")[:, l, :]

            dims = features.shape[1]
            
            X = features

            X_tr, X_te, y_tr, y_te = train_test_split(
                X.numpy(), 
                y, 
                test_size=0.2, 
                random_state=seed, 
                stratify=y)
            
            data_by_layer[l] = (dims, X_tr, X_te, y_tr, y_te)

    elif model_name == "igpt":

        n_layers = [4, 24, 44]

        for l in sorted(n_layers):
            fname = f"clamp_{model_name}_features_{dataset_name}.pt"
            features_path = os.path.join(ds_dir, fname)
            features = torch.load(features_path, map_location="cpu")[:, l, :]

            dims = features.shape[1]

            X = features

            X_tr, X_te, y_tr, y_te = train_test_split(
                X.numpy(), 
                y, 
                test_size=0.2, 
                random_state=seed, 
                stratify=y)
            
            data_by_layer[l] = (dims, X_tr, X_te, y_tr, y_te)

    return n_layers, data_by_layer, classes


def train_classifier(
                    models=models,
                    datasets=datasets,
                    epochs=500,
                    prepare_data=prepare_data,
                    lr=1e-4,
                    device=device,
                    classifier_path=classifier_path,
                    plot=True,
                    scientific=scientific,
                    ):
    

    acc_table_dtd = {mn: {} for mn in models.keys()}
    acc_table = {mn: {} for mn in models.keys()}

    #prepare features per model
    for mr, (model_name, fancy_model_name) in zip(models_roots, models.items()):

        if scientific:

            n_layers, data_by_layer, classes = prepare_data(mr, "dtd")
            
            heads = {}
            for l in n_layers:

                dims, X_tr, X_te, y_tr, y_te = data_by_layer[l]

                X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
                y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)

                linear_classifier = nn.Linear(dims, classes).to(device)
                optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=lr)
                    
                for e in range(epochs):
                    linear_classifier.train()
                    
                    logits = linear_classifier(X_tr)
                    loss = F.cross_entropy(logits, y_tr)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                heads[l] = linear_classifier

            for dataset_name, fancy_dataset_name in datasets.items():
                    
                if dataset_name == "imagenet_val":
                    continue

                n_layers, data_by_layer, classes = prepare_data(mr, dataset_name)

                accuracies = []
                for l in n_layers:

                    dims, X_tr, X_te, y_tr, y_te = data_by_layer[l]

                    X_te = torch.tensor(X_te, dtype=torch.float32, device=device)
                    y_te = torch.tensor(y_te, dtype=torch.long, device=device)

                    linear_classifier = heads[l]
                    linear_classifier.eval()
                    with torch.no_grad():
                        pred = linear_classifier(X_te).argmax(dim=1)
                        acc = (pred == y_te).float().mean().item()

                    accuracies.append(acc)
                    print(f"epoch {e}: test_acc = {acc}")

                    fname = f"{model_name}_linear_classifier_dtd_{dataset_name}_l_{l}.pt"
                    torch.save(linear_classifier, os.path.join(classifier_path, fname))
                    print(f"linear classifier saved: {fname}")

                acc_table_dtd[model_name][dataset_name] = (n_layers, accuracies)

            rows = []
            for model_name, ds_dict in acc_table_dtd.items():
                for dataset_name, (layers, accs) in ds_dict.items():
                    for l, acc in zip(layers, accs):
                        rows.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "layer": l,
                            "accuracy": acc,
                        })

            df = pd.DataFrame(rows)  
            out_dir = os.path.join(csv_path, "csv_scientific_probe")
            out_file = os.path.join(out_dir, f"scientific_probe.csv")
            df.to_csv(out_file, index=False)

        else:
            
            for dataset_name, fancy_dataset_name in datasets.items():

                if dataset_name == "imagenet_val":
                    continue

                n_layers, data_by_layer, classes = prepare_data(mr, dataset_name)

                accuracies = []
                for l in n_layers:

                    dims, X_tr, X_te, y_tr, y_te = data_by_layer[l]

                    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
                    X_te = torch.tensor(X_te, dtype=torch.float32, device=device)
                    y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)
                    y_te = torch.tensor(y_te, dtype=torch.long, device=device)

                    linear_classifier = nn.Linear(dims, classes).to(device)
                    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=lr)

                    for e in range(epochs):
                        linear_classifier.train()
                        
                        logits = linear_classifier(X_tr)
                        loss = F.cross_entropy(logits, y_tr)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    linear_classifier.eval()
                        
                    with torch.no_grad():
                        pred = linear_classifier(X_te).argmax(dim=1)
                        acc = (pred == y_te).float().mean().item()

                    print(f"epoch {e}: test_acc = {acc}")
                    accuracies.append(acc)

                    fname = f"{model_name}_linear_classifier_{dataset_name}_l_{l}.pt"
                    torch.save(linear_classifier, os.path.join(classifier_path, fname))
                    print(f"linear classifier saved: {fname}")

                acc_table[model_name][dataset_name] = (n_layers, accuracies)

            rows = []
            for model_name, ds_dict in acc_table.items():
                for dataset_name, (layers, accs) in ds_dict.items():
                    for l, acc in zip(layers, accs):
                        rows.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "layer": l,
                            "accuracy": acc,
                        })

            df = pd.DataFrame(rows)  
            out_dir = os.path.join(csv_path, "csv_analytical_probe")
            out_file = os.path.join(out_dir, f"analytical_probe.csv")
            df.to_csv(out_file, index=False)


    return None

train_classifier()
