import sys
import os
import omegaconf
import torch
from pathlib import Path

# Get the absolute path to the oned_tokenizer directory
script_dir = Path(__file__).parent  # Directory of your current script
oned_tokenizer_dir = script_dir / "oned_tokenizer"

# Add the oned_tokenizer directory to sys.path
sys.path.insert(0, str(oned_tokenizer_dir))

# Now import the class (use the original repo's import structure)
from modeling.tatitok import TATiTok  # Works because oned_tokenizer is in sys.path


def get_titok_tokenizer():
    folder_path = Path(oned_tokenizer_dir) / "tatitok_bl32_vq_run1"
    config_path = folder_path / "config.yaml"

    config = omegaconf.OmegaConf.load(config_path)
# Use the class
    model = TATiTok(config)

    weights_path = folder_path / "checkpoint-25000/unwrapped_model" / "pytorch_model.bin"
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# _, out_dict = model.encode(torch.randn(1, 3, 64, 64))

# print(out_dict["min_encoding_indices"])
# print(out_dict["min_encoding_indices"].shape)


