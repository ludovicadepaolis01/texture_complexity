import torch
import torchvision.transforms as transforms
import torchvision.utils as u
from PIL import Image
import os
import pickle
import numpy as np
from numba import njit

#code to preprocess images and generate textures with the V&C algorithm

resize = 256

#input paths
img_path = "/your/DTD/path"
img_dir = os.listdir(img_path)

#output paths
coord_path = "/your/texture_coords/path"
texture_dict_path = "/your/texture_dictionary/path"
save_bool = "/your/binary_images/path"

for d in [coord_path, texture_dict_path, save_bool]:
    os.makedirs(d, exist_ok=True)

#preprocess images: make boolean
def make_bool_images(
        img_dir = img_dir,
        texture_dict_path = texture_dict_path,
        save_bool = save_bool,
        ):
    
    img_dict = {}
    for texture_class in img_dir:
        class_path = os.path.join(img_path, texture_class)
        if os.path.isdir(class_path):
            img_list = []
            for img in os.listdir(class_path):
                file_path = os.path.join(class_path, img)
                img_list.append(file_path)
                img_dict[texture_class] = img_list
                with open(os.path.join(texture_dict_path, "texture_dict.pkl"), "wb") as f:
                    pickle.dump(img_dict, f)  
                            
                for orig_image in img_list:
                    file_path = os.path.join(img_path, orig_image)

                    image = Image.open(file_path).convert("L")
                    image = image.resize((resize, resize))

                    np_image = np.array(image) #[H, W]

                    median = np.median(np_image) #median threshold
                    bool_image = (np_image > median).astype(np.bool_) #[H, W] boolean
                    
                    out = os.path.join(save_bool, f"{os.path.splitext(os.path.basename(file_path))[0]}.npy")
                    np.save(out, bool_image)

                print(len(img_list))

        print(texture_class, bool_image.shape, bool_image.dtype, bool_image.mean())


make_bool_images()

#generate textures
@njit
def histogram(X):
    # X should be an array of bools!
    
    height, width = X.shape
    p = np.zeros(16, dtype=np.float64)
    
    for i in range(height - 1):
        for j in range(width - 1):
            # define a shorthand notation for the four pixels in the glider
            pix1 = X[i, j]
            pix2 = X[i, j + 1]
            pix3 = X[i + 1, j]
            pix4 = X[i + 1, j + 1]

            # figure out what is the state of the glider in the current position.
            # the state is just the integer representation of the glider seen as a binary string.
            # so if the glider is [pix1 pix2; pix3 pix4], the state is 8*pix1+4*pix2+2*pix3+pix4.
            ind = (pix1 << 3) | (pix2 << 2) | (pix3 << 1) | pix4

            # increment by one the count of the occurrences of the state we have found
            p[ind] += 1.0

    total = p.sum()
    if total > 0:
        p /= total
    return p

@njit
def fourier_transform(hist, state):
    # hist is an array of length 16 containing the frequency of each state in the spin representation,
    # in other words p([A1 A2; A3 A4]) used in eq 7 in V&C2012 corresponds to hist[8*A1+4*A2+2*A3+A4]
    #
    # state should be an array-like of bools of size (4,), indicating [s1 s2 s3 s4] in eq 7 in V&C2012
    s1, s2, s3, s4 = state

    
    transform = 0
    for A1 in [0,1]:
        for A2 in [0,1]:
            for A3 in [0,1]:
                for A4 in [0,1]:

                    ind = (A1 << 3) | (A2 << 2) | (A3 << 1) | A4

                    transform += hist[ind] * (-1) ** (A1*s1 + A2*s2 + A3*s3 + A4*s4)

    return transform 

@njit
def texture_coords(hist):
    # returns the (reduced) fourier transform of the given histogram as a size-10 array,
    # specified in terms of the coordinates [γ, β—, β|, β\, β/, θ◢, θ◤, θ◥, θ◣, α].
    # in terms of the coordinate names used in metex, this is [gamma, beta1, beta2, beta3, beta4, theta1, theta2, theta3, theta4, alpha]

    transform = np.zeros(10)

    # for the definitions, see eqs 13-16 in V&C2012

    # gamma (note the minus sign! as per convention in V&C)
    transform[0] = -fourier_transform(hist, [1,0,0,0])

    # beta1
    transform[1] = fourier_transform(hist, [1,1,0,0])

    # beta2
    transform[2] = fourier_transform(hist, [1,0,1,0])

    # beta3
    transform[3] = fourier_transform(hist, [1,0,0,1])

    # beta4
    transform[4] = fourier_transform(hist, [0,1,1,0])

    # theta1 (note the minus sign!)
    transform[5] = -fourier_transform(hist, [0,1,1,1])

    # theta2
    transform[6] = -fourier_transform(hist, [1,1,1,0])

    # theta3
    transform[7] = -fourier_transform(hist, [1,1,0,1])

    # theta4
    transform[8] = -fourier_transform(hist, [1,0,1,1])

    # alpha
    transform[9] = fourier_transform(hist, [1,1,1,1])

    return transform

def texture_coords_from_image(X):
    X = np.array(X, dtype=np.bool_)

    return texture_coords(histogram(X))

bool_files = []
for f in os.listdir(save_bool):
    if f.endswith(".npy"):
        bool_files.append(f)

# compute Victor coords for each image and store results
coords_dict = {}  # filename -> 10 coords
coords_list = []  # list of coords for easy stacking later

for fname in bool_files:
    fpath = os.path.join(save_bool, fname)   # FULL PATH
    X = np.load(fpath)                       # [H,W] bool
    c = texture_coords_from_image(X)         # (10,)
    coords_dict[fname] = c
    coords_list.append(c)

coords_array = np.stack(coords_list, axis=0)  # [N,10]
print("Computed coords:", coords_array.shape)

out_pkl = os.path.join(coord_path, "victor_coordinates.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump(coords_dict, f)

out_npy = os.path.join(coord_path, "victor_coordinates.npy")
np.save(out_npy, coords_array)
