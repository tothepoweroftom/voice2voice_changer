import os

import torch
from const import UPLOAD_DIR
from RVC.modelMerger.MergeModel import merge_model
from utils.ModelMerger import ModelMerger, ModelMergerRequest
from utils.VoiceChangerParams import VoiceChangerParams


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, params: VoiceChangerParams, request: ModelMergerRequest, storeSlot: int):
        merged = merge_model(params, request)

        # Once done, save it to the upload folder. (Historical background)
        # It can be moved to the persistent model folder by calling subsequent loadmodel.
        storeDir = os.path.join(UPLOAD_DIR)
        print("[Voice Changer] store merged model to:", storeDir)
        os.makedirs(storeDir, exist_ok=True)
        storeFile = os.path.join(storeDir, "merged.pth")
        torch.save(merged, storeFile)
        return storeFile
