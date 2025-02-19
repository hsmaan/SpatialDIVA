from PIL import Image
import torch
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def load_uni_model():
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform
    
def convert_arr_pil(arr, transform):
    pil_image = Image.fromarray(arr)
    pil_image = transform(pil_image).unsqueeze(dim=0)
    return pil_image

def uni_feature_extract(image, model):
    with torch.inference_mode():
        feature_emb = model(image)
        return feature_emb

