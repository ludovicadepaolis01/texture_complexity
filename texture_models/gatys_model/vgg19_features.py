import torch
import torch.nn as nn
import torchvision.models as models

vgg19_pretrained = models.vgg19_bn()
print(vgg19_pretrained)

OMP_NUM_THREADS=1

layer_indices = [1, 4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41, 44, 47, 50]

#VGG19
class VGG19_features(nn.Module):
    model_path = "your/path/to/vgg19_bn"
    def __init__(self, selected_idx: int):
        super().__init__()
        self.selected_idx = int(selected_idx)
        self.vgg19_pretrained = models.vgg19_bn()
        state_dict = torch.load(self.model_path)
        self.vgg19_pretrained.load_state_dict(state_dict)
        self.vgg19_pretrained.to("cuda").eval()
        self.vgg19_pretrained.requires_grad_(False)
        
        self.feature_map = None

        layer = self.vgg19_pretrained.features[self.selected_idx]
        if isinstance(layer, torch.nn.BatchNorm2d): 
            layer._hook_idx = self.selected_idx
            hook = layer.register_forward_hook(self.hook_func)
            layer.name = f'layer_{self.selected_idx}'
            print(layer.name)

    def hook_func(self, 
                  module, 
                  input, 
                  output):
        self.feature_map = output.detach().cpu()
    
    def forward(self, 
                images,):

        self.feature_map = None
        #dummy forward
        _ = self.vgg19_pretrained(images)

        return self.feature_map
