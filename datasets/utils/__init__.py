import os
import inspect
import importlib
from datasets.utils.federated_dataset import FederatedDataset
from argparse import Namespace


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('datasets')
            if not model.find('__') > -1 and 'py' in model and model.split('.')[0] != '']


NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('datasets.' + model)
    dataset_classes_name = [x for x in mod.__dir__() if
                            'type' in str(type(getattr(mod, x))) and 'ContinualDataset' in str(
                                inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c

    gcl_dataset_classes_name = [x for x in mod.__dir__() if
                                'type' in str(type(getattr(mod, x))) and 'GCLDataset' in str(
                                    inspect.getmro(getattr(mod, x))[1:])]
    for d in gcl_dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c


def get_dataset(args: Namespace) -> FederatedDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)