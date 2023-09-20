"""Base model class from which all subclasses for downstream models will inherit from."""

from abc import ABC, abstractmethod


class HuggingFaceModel(ABC):

    @property
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
