from Exceptions import WeightDownladException
from downloader.SampleDownloader import downloadInitialSamples
from downloader.WeightDownloader import downloadWeight
import argparse
from RVC.RVCr2 import RVCr2
from VoiceChangerParamsManager import VoiceChangerParamsManager
from utils.VoiceChangerModel import AudioInOut
from utils.VoiceChangerParams import VoiceChangerParams
from distutils.util import strtobool
from data.ModelSlot import RVCModelSlot
import platform
from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


def printMessage(message, level=0):
    pf = platform.system()
    if pf == "Windows":
        if level == 0:
            message = f"{message}"
        elif level == 1:
            message = f"    {message}"
        elif level == 2:
            message = f"    {message}"
        else:
            message = f"    {message}"
    else:
        if level == 0:
            message = f"\033[17m{message}\033[0m"
        elif level == 1:
            message = f"\033[34m    {message}\033[0m"
        elif level == 2:
            message = f"\033[32m    {message}\033[0m"
        else:
            message = f"\033[47m    {message}\033[0m"
    logger.info(message)

def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model_dir", help="path to model files")
    parser.add_argument("--sample_mode", type=str, default="production", help="rvc_sample_mode")

    parser.add_argument("--content_vec_500", type=str, default="pretrain/checkpoint_best_legacy_500.pt", help="path to content_vec_500 model(pytorch)")
    parser.add_argument("--content_vec_500_onnx", type=str, default="pretrain/content_vec_500.onnx", help="path to content_vec_500 model(onnx)")
    parser.add_argument("--content_vec_500_onnx_on", type=strtobool, default=True, help="use or not onnx for  content_vec_500")
    parser.add_argument("--hubert_base", type=str, default="pretrain/hubert_base.pt", help="path to hubert_base model(pytorch)")
    parser.add_argument("--hubert_base_jp", type=str, default="pretrain/rinna_hubert_base_jp.pt", help="path to hubert_base_jp model(pytorch)")
    parser.add_argument("--hubert_soft", type=str, default="pretrain/hubert/hubert-soft-0d54a1f4.pt", help="path to hubert_soft model(pytorch)")
    parser.add_argument("--nsf_hifigan", type=str, default="pretrain/nsf_hifigan/model", help="path to nsf_hifigan model(pytorch)")
    parser.add_argument("--crepe_onnx_full", type=str, default="pretrain/crepe_onnx_full.onnx", help="path to crepe_onnx_full")
    parser.add_argument("--crepe_onnx_tiny", type=str, default="pretrain/crepe_onnx_tiny.onnx", help="path to crepe_onnx_tiny")
    parser.add_argument("--rmvpe", type=str, default="pretrain/rmvpe.pt", help="path to rmvpe")
    parser.add_argument("--rmvpe_onnx", type=str, default="pretrain/rmvpe.onnx", help="path to rmvpe onnx")
    parser.add_argument("--audio_file", type=str, default="test_1.wav", help="Path to input audio file")

    return parser

parser = setupArgParser()
args, unknown = parser.parse_known_args()

voiceChangerParams = VoiceChangerParams(
    model_dir=args.model_dir,
    content_vec_500=args.content_vec_500,
    content_vec_500_onnx=args.content_vec_500_onnx,
    content_vec_500_onnx_on=args.content_vec_500_onnx_on,
    hubert_base=args.hubert_base,
    hubert_base_jp=args.hubert_base_jp,
    hubert_soft=args.hubert_soft,
    nsf_hifigan=args.nsf_hifigan,
    crepe_onnx_full=args.crepe_onnx_full,
    crepe_onnx_tiny=args.crepe_onnx_tiny,
    rmvpe=args.rmvpe,
    rmvpe_onnx=args.rmvpe_onnx,
    sample_mode=args.sample_mode,
)
vcparams = VoiceChangerParamsManager.get_instance()
vcparams.setParams(voiceChangerParams)

if __name__ == "__main__":
    # Download weights
    try:
        downloadWeight(voiceChangerParams)
    except WeightDownladException:
        printMessage("failed to download weight for rvc", level=2)

    # Download Sample
    try:
        downloadInitialSamples(args.sample_mode, args.model_dir)
    except Exception as e:
        printMessage(f"[Voice Changer] loading sample failed {e}", level=2)

    