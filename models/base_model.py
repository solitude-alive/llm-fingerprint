from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_matrix(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
