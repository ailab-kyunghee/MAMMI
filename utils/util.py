import torch 
import numpy as np

import colorsys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

from models import densenet # resnet, vit

def load_model(check_point, model, device):
    ##### load model
    checkpoint = torch.load(check_point, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        checkpoint_model = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
    model = densenet.__dict__['densenet121'](num_classes=14)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    
    model = model.to(device)
    model.eval()

    return model

def get_alpha_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        c = np.array((cmap[0]/255.0, cmap[1]/255.0, cmap[2]/255.0))

        cmax = colorsys.rgb_to_hls(*c)
        cmax = np.array(cmax)
        cmax[-1] = 1.0

        cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c,cmax])

    alpha_cmap = cmap(np.arange(256))
    alpha_cmap[:,-1] = np.linspace(0, 0.85, 256)
    alpha_cmap = ListedColormap(alpha_cmap)

    return alpha_cmap

def show(img, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min();img /= img.max()

    plt.imshow(img, **kwargs); plt.axis('off')