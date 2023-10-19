import torch
from torch import device
from RVC.embedder.Embedder import Embedder
from fairseq import checkpoint_utils


class FairseqHubert(Embedder):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().setProps("hubert_base", file, dev, isHalf)

        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [file],
            suffix="",
        )
        model = models[0]
        model.eval()

        model = model.to(dev)
        if isHalf:
            model = model.half()

        self.model = model
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        padding_mask = torch.BoolTensor(feats.shape).to(self.dev).fill_(False)

        # In the original_v1, final_proj was applied to L9.(-> 256)
        # The original_v2 does not apply final_proj to L12.(-> 768)

        inputs = {
            "source": feats.to(self.dev),
            "padding_mask": padding_mask,
            "output_layer": embOutputLayer,  # 9 or 12
        }

        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            if useFinalProj:
                feats = self.model.final_proj(logits[0])
            else:
                feats = logits[0]
        return feats
