import torch
import torch.nn as nn
import torchvision.models as models
import einops

OMP_NUM_THREADS=1

#VGG19
class VGG19_representations(nn.Module):
     model_path = "your/path/to/vgg19_bn"
     def __init__(self):
          super().__init__()
          self.vgg19_pretrained = models.vgg19_bn()
          state_dict = torch.load(VGG19_representations.model_path)
          self.vgg19_pretrained.load_state_dict(state_dict)
          self.vgg19_pretrained.to("cuda")
          self.vgg19_pretrained.requires_grad_(False)
          self.hooks = []
          self.feature_maps = {}

          count = 0
          for (idx, layer) in enumerate(self.vgg19_pretrained.features):
               if isinstance(layer, torch.nn.BatchNorm2d):
                    if idx in [1, 8, 15, 28, 41]:
                         hook = layer.register_forward_hook(self.hook_func)
                         layer.name = f'layer_{count}'
                         self.hooks.append(hook)
               count += 1

     def hook_func(self, module, input, output):
          name = module.name
          self.feature_maps[name] = output
          
     def gram_matrix(self, feature_map):
          gram_matrix = torch.einsum("bihw,bjhw->bij", feature_map, feature_map)
          
          return gram_matrix
     
     def forward(self, images):
          gram_matrix_list = []
          feature_map_m_list = []
          feature_map_n_list = []
          feature_map_list = []
          #dummy forward
          _ = self.vgg19_pretrained(images) 

          for key in self.feature_maps:
               feature_map = self.feature_maps[key]
               feature_map_height = feature_map.size(2)
               feature_map_width = feature_map.size(3)
               feature_map_m = feature_map_height*feature_map_width
               feature_map_m_list.append(feature_map_m)
               gram_matrices = self.gram_matrix(feature_map)
               gram_matrix_list.append(gram_matrices)
               feature_map_list.append(feature_map)
               
          return gram_matrix_list, feature_map_m_list
     
def gaussian_image_tensor(size=400, mean=0.5, std=0.2):
    gaussian_image = torch.randn(3, size, size) * std + mean
    gaussian_image.clamp_(0.0, 1.0)
    return gaussian_image.to("cuda")
