import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import compute_module_sizes, get_max_layer_size
from huggingface_hub import HfApi

from .imports import is_transformers_available, is_diffusers_available, is_timm_available

if is_timm_available():
    import timm

if is_transformers_available():
    from transformers import AutoConfig, AutoModel

# TODO: Support diffusers models
if is_diffusers_available():
    # requires weights downloading
    from diffusers import DiffusionPipeline

def get_supported_library(model_name:str):
    "Check the Hub for the model's metadata to determine the library it is supported by."
    api = HfApi()
    model_info = api.model_info(model_name)
    return getattr(model_info, "library_name", False)

def create_empty_model(model_name, library_name:str = None):
    """
    Creates an empty model from its parent library on the `Hub` 
    to calculate the overall memory consumption.

    Args:
        model_name (`str`):
            The model name on the Hub
        library_name (`str`, *optional*, defaults to None):
            The library the model has an integration with, such as `transformers`. Will
            be used if `model_name` has no metadata on the Hub to determine the library.
    """
    if library_name is None:
        library_name = get_supported_library(model_name)
        if library_name == False:
            raise ValueError(f"Model {model_name} does not have any library metadata on the Hub, please manually pass in a `library_name` to use (such as `transformers`)")
    if library_name == "transformers":
        if not is_transformers_available():
            raise ImportError(f"To check `{model_name}`, `transformers` must be installed. Please install it via `pip install transformers`")
        print(f'Loading pretrained config for `{model_name}` from `transformers`...')
        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            model = AutoModel.from_config(config)
    elif library_name == "timm":
        if not is_timm_available():
            raise ImportError(f"To check `{model_name}`, `timm` must be installed. Please install it via `pip install timm`")
        print(f'Loading pretrained config for `{model_name}` from `timm`...')
        with init_empty_weights():
            model = timm.create_model(model_name, pretrained=False)
    else:
        raise ValueError(f"Library `{library_name}` is not supported, please open an issue on GitHub for us to add support for it")
    return model

def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f'{round(size, 2)} {x}'
        size /= 1024.0

    return size

def get_sizes(model:torch.nn.Module):
    sizes = compute_module_sizes(model)
    modules_to_treat = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )
    largest_layer = get_max_layer_size(modules_to_treat, sizes, [])
    total_size = sizes['']
    return total_size, largest_layer