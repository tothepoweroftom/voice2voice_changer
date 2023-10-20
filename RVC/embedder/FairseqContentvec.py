from torch import device
from RVC.embedder.Embedder import Embedder
from RVC.embedder.FairseqHubert import FairseqHubert


class FairseqContentvec(FairseqHubert):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().loadModel(file, dev, isHalf)
        super().setProps("contentvec", file, dev, isHalf)
        return self
