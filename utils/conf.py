import random
import tensorflow as tf
import numpy as np


def get_device(device_id: int = 0) -> str:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"/GPU:{device_id}"
    else:
        return "/CPU:0"


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
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
