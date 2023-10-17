from typing import Protocol
from const import PitchExtractorType
from pitchExtractor.CrepeOnnxPitchExtractor import CrepeOnnxPitchExtractor
from pitchExtractor.DioPitchExtractor import DioPitchExtractor
from pitchExtractor.HarvestPitchExtractor import HarvestPitchExtractor
from pitchExtractor.CrepePitchExtractor import CrepePitchExtractor
from pitchExtractor.PitchExtractor import PitchExtractor
from pitchExtractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor
from pitchExtractor.RMVPEPitchExtractor import RMVPEPitchExtractor
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
