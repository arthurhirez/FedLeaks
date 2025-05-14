from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Tuple, List

import numpy as np


class FederatedDataset(ABC):
    NAME = None
    SETTING = None
    N_SAMPLES_PER_CLASS = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders: List = []
        self.test_loader: List = []
        self.args = args

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[np.array, np.array]:
        """
        Should return lists of np.Tensors.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num: int, names_list: List[str], n_series : int):
        """
        Returns the backbone model (Keras Model).
        """
        pass

    # @staticmethod
    # @abstractmethod
    # def get_transform()# -> tf.keras.Sequential:
    #     """
    #     Returns a tf.keras Sequential model representing data preprocessing.
    #     """
    #     pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform():
        """
        Returns a tf.keras layer or function for normalization.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform():
        """
        Returns a tf.keras layer or function for denormalization.
        """
        pass

    # @staticmethod
    # @abstractmethod
    # def get_scheduler(model: Model, args: Namespace) -> LearningRateSchedule:
    #     """
    #     Returns a learning rate schedule for the optimizer.
    #     """
    #     pass

    @staticmethod
    def get_epochs() -> int:
        pass

    @staticmethod
    def get_batch_size() -> int:
        pass
