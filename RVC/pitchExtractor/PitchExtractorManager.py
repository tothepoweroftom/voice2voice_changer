from typing import Protocol
from const import PitchExtractorType
from RVC.pitchExtractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
from RVC.pitchExtractor.DioPitchExtractor import DioPitchExtractor
from RVC.pitchExtractor.HarvestPitchExtractor import HarvestPitchExtractor
from RVC.pitchExtractor.CrepePitchExtractor import CrepePitchExtractor
from RVC.pitchExtractor.PitchExtractor import PitchExtractor
from RVC.pitchExtractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor
from RVC.pitchExtractor.RMVPEPitchExtractor import RMVPEPitchExtractor
from utils.VoiceChangerParams import VoiceChangerParams


class PitchExtractorManager(Protocol):
    currentPitchExtractor: PitchExtractor | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        cls.currentPitchExtractor = cls.loadPitchExtractor(pitchExtractorType,  gpu)
        return cls.currentPitchExtractor

    @classmethod
    def loadPitchExtractor(
        cls, pitchExtractorType: PitchExtractorType, gpu: int
    ) -> PitchExtractor:
        if pitchExtractorType == "harvest":
            return HarvestPitchExtractor()
        elif pitchExtractorType == "dio":
            return DioPitchExtractor()
        elif pitchExtractorType == "crepe":
            return CrepePitchExtractor(gpu)
        elif pitchExtractorType == "crepe_tiny":
            return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_tiny, gpu)
        elif pitchExtractorType == "crepe_full":
            return CrepeOnnxPitchExtractor(pitchExtractorType, cls.params.crepe_onnx_full, gpu)
        elif pitchExtractorType == "rmvpe":
            return RMVPEPitchExtractor(cls.params.rmvpe, gpu)
        elif pitchExtractorType == "rmvpe_onnx":
            return RMVPEOnnxPitchExtractor(cls.params.rmvpe_onnx, gpu)
        else:
            # return hubert as default
            print("[Voice Changer] PitchExctractor not found", pitchExtractorType)
            print("                fallback to dio")
            return DioPitchExtractor()
