from typing import Protocol

from utils.LoadModelParams import LoadModelParams


class ModelSlotGenerator(Protocol):
    @classmethod
    def loadModel(cls, params: LoadModelParams):
        ...
