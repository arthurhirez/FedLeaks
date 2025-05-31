import random
import torch
import numpy as np


def get_device(device_id: int = 0) -> torch.device:
    return torch.device("cpu")
    
    # if torch.cuda.is_available():
    #     return torch.device(f"cuda:{device_id}")
    # else:
    #     return torch.device("cpu")


def data_path() -> str:
    # return 'F://dataset/pic_cls/'
    return "C:/Users/arthu/USPy/0_BEPE/0_TSWater/FedLeakages/datasets/"

def base_path() -> str:
    # return './data/'
    return "C:/Users/arthu/USPy/0_BEPE/0_TSWater/FedLeakages/data_leaks/"


def checkpoint_path() -> str:
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Optional for deterministic behavior (may slow down training)
    torch.use_deterministic_algorithms(True)
