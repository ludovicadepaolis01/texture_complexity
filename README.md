This is the code for "Vision Transformers and humans align in representing texture complexity" by de Paolis et al.

- **Requirements**
  ```text
  torch = 2.2.0a0+git6c8c5ad
  torchvision = 0.17.0+b2383d4
  transformers = 4.38.0
  huggingface-hub = 0.22.1
  tokenizers = 0.15.2
  safetensors = 0.4.2
  numpy = 1.26.4
  scipy = 1.12.0
  pandas = 2.2.1
  matplotlib = 3.8.3
  Pillow (PIL) = 10.2.0
  scikit-learn = 1.4.1.post1
  
- **Source data**  
  In `/data` you can find the following two source images:  
      ```text
  `braided_0070.jpg` from _Describable Textures Dataset_ (**DTD**) in **[Describing Textures in the Wild (Cimpoi et al., 2014)](https://arxiv.org/abs/1311.3618)** to generate synthetic textures  
  `ILSVRC2012_val_00000256.JPEG` from _ImageNet_ (**Object**) in **[ImageNet Large Scale Visual Recognition Challenge (Russakovsky et al., 2015)](https://arxiv.org/abs/1409.0575)** as object image  
  
- **Models**  
  Models are available on Torchvision and HuggingFace:  
    ```text
  CLIP = pytorch_model.bin / ViT-B-32.pt
  DINO-v2 = model.safetensors
  iGPT = pytorch_model.bin
  VGG19 = vgg19_bn-c79401a0.pth
  
- **Data preprocess**  
  Preprocess data in `/data` by running `/data_preprocess/data_preprocess.py`.  
  
- **Texture generation**  
  Generate Noise images, and V&C, P&S, G type textures by running `/texture_synthesis/generate_textures.py`.  
  This code uses `/texture_synthesis/define_algorithms.py` and the models' source codes in  respectively: `/texture_models/victor_model`, `/texture_models/portilla_model`, `/texture_models/gatys_model`.  

- **Features extraction**  
   Extract VGG19 features by running `/src/vgg_feature_extraction.py`.  
   Extract ViTs features by running `/src/vits_feature_extraction.py`.  
   Preprocess by running `/src/preprocess_features.py` by performing clamping (`/src/clamp_features.py`).  
  
- **Analyses**  
   Perform all II analyses by running `/analyses/ii_builders.py`.  
   Perform linear all classifier analyses by running `/analyses/linear_probe.py`.  
   Perform human data analysis and correlations with **across datasets II** data by running `/analyses/behavioral_analyses.ipynb`.  
  
