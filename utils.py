import torch
import rioxarray as rxr
from terratorch.registry import FULL_MODEL_REGISTRY

def tif_to_tensor(filename: str):
    tif = rxr.open_rasterio(filename)
    # Convert to shape [B, C, 224, 224]
    image_tensor = torch.Tensor(tif.values, device='cpu').unsqueeze(0)
    return image_tensor

def build_model_from_registry(
        name: str = 'terramind_v1_base_generate',
        input_modalities: list = ['S2L2A'],
        output_modalities: list = ['LULC'],
        pretrained: bool = True,
        standartize: bool = True,
        ):
    model = FULL_MODEL_REGISTRY.build(
    name,
    modalities=input_modalities,
    output_modalities=output_modalities,
    pretrained=pretrained,
    standardize=standartize,
    )

    return model