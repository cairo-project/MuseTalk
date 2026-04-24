"""
MuseTalk: Audio-driven lip-sync portrait animation.

Public API:
  load_models(...)           -> dict  – load all model weights onto GPU
  generate_video(models, ref_path, audio_path, save_path, **kwargs) -> str
                                      – synthesise a lip-synced video from a
                                        reference image, video, or image directory
  generate(args)             -> str   – convenience: load_models + generate_video

CLI:
  python generate_musetalk.py --ref_image face.jpg --audio speech.wav --output out.mp4
  python generate_musetalk.py --ref_video face.mp4 --audio speech.wav --output out.mp4

Note: importing this module changes the working directory to the MuseTalk repo root
so that the mmpose / face-detection models (initialised at module level in
musetalk/utils/preprocessing.py) can find their config and checkpoint files.
"""

import argparse
import copy
import glob
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

# Must happen before musetalk imports: preprocessing.py initialises mmpose/face-detection
# at module level using paths relative to the MuseTalk repo root.
_MUSETALK_ROOT = Path(__file__).resolve().parent
os.chdir(_MUSETALK_ROOT)
if str(_MUSETALK_ROOT) not in sys.path:
    sys.path.insert(0, str(_MUSETALK_ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen, get_file_type, get_video_fps

_DEFAULT_MODEL_DIR = os.environ.get("MUSETALK_MODEL_DIR", "/fsx/shared/users/landz/models/Musetalk")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _build_model_paths(model_dir: str, version: str) -> dict:
    if version == "v15":
        unet_subdir = "musetalkV15"
        unet_filename = "unet.pth"
    else:
        unet_subdir = "musetalk"
        unet_filename = "pytorch_model.bin"
    return {
        "unet_model_path": os.path.join(model_dir, unet_subdir, unet_filename),
        "unet_config": os.path.join(model_dir, unet_subdir, "musetalk.json"),
        "vae_model_path": os.path.join(model_dir, "sd-vae"),
        "whisper_dir": os.path.join(model_dir, "whisper"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_models(
    model_dir: str = None,
    version: str = "v15",
    gpu_id: int = 0,
    use_float16: bool = True,
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
) -> dict:
    """Load all MuseTalk model weights and return a model bundle dict.

    Args:
        model_dir:   Directory containing the MuseTalk model subdirectories.
                     Defaults to $MUSETALK_MODEL_DIR or /fsx/.../Musetalk.
        version:     ``"v15"`` (default) or ``"v1"``.
        gpu_id:      CUDA device index.
        use_float16: Run models in half precision.
        left_cheek_width / right_cheek_width: v1.5 face-blending parameters.

    Returns:
        dict with keys: vae, unet, pe, audio_processor, whisper, fp, device,
        use_float16, version, weight_dtype.
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    paths = _build_model_paths(model_dir, version)

    print(f"Loading VAE from {paths['vae_model_path']}")
    from musetalk.models.vae import VAE
    vae = VAE(model_path=paths["vae_model_path"])

    print(f"Loading UNet from {paths['unet_model_path']}")
    from musetalk.models.unet import UNet, PositionalEncoding
    unet = UNet(
        unet_config=paths["unet_config"],
        model_path=paths["unet_model_path"],
        device=device,
    )
    pe = PositionalEncoding(d_model=384)

    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    print(f"Loading Whisper from {paths['whisper_dir']}")
    audio_processor = AudioProcessor(feature_extractor_path=paths["whisper_dir"])
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(paths["whisper_dir"])
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    print("Loading FaceParsing model")
    if version == "v15":
        fp = FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
        )
    else:
        fp = FaceParsing()

    print("All models loaded.")
    return {
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "audio_processor": audio_processor,
        "whisper": whisper,
        "fp": fp,
        "device": device,
        "use_float16": use_float16,
        "version": version,
        "weight_dtype": weight_dtype,
    }


@torch.no_grad()
def generate_video(
    models: dict,
    ref_path: str,
    audio_path: str,
    save_path: str,
    *,
    bbox_shift: int = 0,
    fps: int = 25,
    batch_size: int = 8,
    extra_margin: int = 10,
    audio_padding_length_left: int = 2,
    audio_padding_length_right: int = 2,
    parsing_mode: str = "jaw",
    use_saved_coord: bool = False,
    coord_cache_path: str = None,
    save_coord: bool = False,
) -> str:
    """Generate a lip-synced video from a reference image/video and an audio file.

    Args:
        models:       Bundle returned by ``load_models()``.
        ref_path:     Path to a source image (.jpg/.png), video (.mp4/.mov/…),
                      or directory of sequentially-named images.
        audio_path:   Path to audio file (wav/mp3/mp4/…).
        save_path:    Where to write the output .mp4.
        bbox_shift:   Bounding-box shift (v1 only; ignored for v15).
        fps:          Output frame rate — used when ref is an image or directory.
        batch_size:   UNet inference batch size.
        extra_margin: Extra pixels added below the face bbox (v15 only).
        audio_padding_length_left / right: Whisper chunk context padding.
        parsing_mode: BiSeNet blending mode (``"jaw"``, ``"neck"``, or ``"raw"``).
        use_saved_coord: Load cached landmarks from ``coord_cache_path``.
        coord_cache_path: Path to landmark cache file (.pkl).
        save_coord:   Persist computed landmarks to ``coord_cache_path``.

    Returns:
        Absolute path to the saved video (== ``save_path``).
    """
    vae = models["vae"]
    unet = models["unet"]
    pe = models["pe"]
    audio_processor = models["audio_processor"]
    whisper = models["whisper"]
    fp = models["fp"]
    device = models["device"]
    version = models["version"]
    weight_dtype = models["weight_dtype"]

    if version == "v15":
        bbox_shift = 0

    timesteps = torch.tensor([0], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Extract frames ---
        file_type = get_file_type(ref_path)
        if file_type == "video":
            frames_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frames_dir)
            os.system(
                f"ffmpeg -v fatal -i {ref_path} -start_number 0 {frames_dir}/%08d.png"
            )
            input_img_list = sorted(
                glob.glob(os.path.join(frames_dir, "*.[jpJP][pnPN]*[gG]"))
            )
            fps = get_video_fps(ref_path)
        elif file_type == "image":
            input_img_list = [ref_path]
        elif os.path.isdir(ref_path):
            input_img_list = sorted(
                glob.glob(os.path.join(ref_path, "*.[jpJP][pnPN]*[gG]")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )
        else:
            raise ValueError(
                f"{ref_path} must be a video file, image file, or image directory"
            )

        # --- Audio features ---
        print("Extracting audio features...")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )

        # --- Landmarks / bounding boxes ---
        if coord_cache_path and os.path.exists(coord_cache_path) and use_saved_coord:
            print("Using cached coordinates")
            with open(coord_cache_path, "rb") as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("Extracting face landmarks (this may take a while)...")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            if save_coord and coord_cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(coord_cache_path)), exist_ok=True)
                with open(coord_cache_path, "wb") as f:
                    pickle.dump(coord_list, f)

        print(f"Reference frames: {len(frame_list)}")

        # --- VAE encode face crops ---
        print("Encoding face crops...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if version == "v15":
                y2 = min(y2 + extra_margin, frame.shape[0])
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            input_latent_list.append(vae.get_latents_for_unet(crop))

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # --- UNet inference ---
        print("Running UNet inference...")
        video_num = len(whisper_chunks)
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device,
        )
        res_frame_list = []
        total = int(np.ceil(float(video_num) / batch_size))
        for whisper_batch, latent_batch in tqdm(gen, total=total, desc="Inference"):
            audio_feature_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = vae.decode_latents(pred_latents)
            res_frame_list.extend(recon)

        # --- Composite generated faces back onto original frames ---
        print("Compositing frames...")
        result_frames_dir = os.path.join(tmpdir, "result_frames")
        os.makedirs(result_frames_dir)
        for i, res_frame in enumerate(tqdm(res_frame_list, desc="Compositing")):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            if version == "v15":
                y2 = min(y2 + extra_margin, ori_frame.shape[0])
            try:
                res_frame_resized = cv2.resize(
                    res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                )
            except Exception:
                continue
            if version == "v15":
                combine_frame = get_image(
                    ori_frame, res_frame_resized, [x1, y1, x2, y2],
                    mode=parsing_mode, fp=fp,
                )
            else:
                combine_frame = get_image(
                    ori_frame, res_frame_resized, [x1, y1, x2, y2], fp=fp,
                )
            cv2.imwrite(
                os.path.join(result_frames_dir, f"{str(i).zfill(8)}.png"),
                combine_frame,
            )

        # --- Encode to video and mux audio ---
        print("Encoding output video...")
        out_dir = os.path.dirname(os.path.abspath(save_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        temp_vid = os.path.join(tmpdir, "temp_silent.mp4")
        os.system(
            f"ffmpeg -y -v warning -r {fps} -f image2 "
            f"-i {result_frames_dir}/%08d.png "
            f"-vcodec libx264 "
            f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p' "
            f"-crf 18 {temp_vid}"
        )
        os.system(
            f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid} {save_path}"
        )

    print(f"Saved to {save_path}")
    return os.path.abspath(save_path)


def generate(args) -> str:
    """Convenience wrapper: load models then generate one video.

    Accepts a namespace with attributes matching the CLI arguments.
    Either ``args.ref_image`` or ``args.ref_video`` must be set.
    """
    ref_path = args.ref_image or args.ref_video
    if not ref_path:
        raise ValueError("Provide either --ref_image or --ref_video")

    models = load_models(
        model_dir=args.model_dir,
        version=args.version,
        gpu_id=args.gpu_id,
        use_float16=args.use_float16,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )
    return generate_video(
        models=models,
        ref_path=ref_path,
        audio_path=args.audio,
        save_path=args.output,
        bbox_shift=args.bbox_shift,
        fps=args.fps,
        batch_size=args.batch_size,
        extra_margin=args.extra_margin,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
        parsing_mode=args.parsing_mode,
        use_saved_coord=args.use_saved_coord,
        coord_cache_path=args.coord_cache,
        save_coord=args.save_coord,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuseTalk lip-sync inference")

    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref_image", type=str, default=None,
                           help="Reference image (.jpg/.png)")
    ref_group.add_argument("--ref_video", type=str, default=None,
                           help="Reference video (.mp4/…) or directory of images")

    parser.add_argument("--audio", type=str, required=True,
                        help="Audio file (wav/mp3/mp4)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video path (.mp4)")
    parser.add_argument("--model_dir", type=str, default=_DEFAULT_MODEL_DIR,
                        help="Directory containing MuseTalk model subdirectories")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"],
                        help="Model version")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--use_float16", action="store_true", default=True,
                        help="Use float16 (default: on)")
    parser.add_argument("--no_float16", dest="use_float16", action="store_false",
                        help="Disable float16")
    parser.add_argument("--bbox_shift", type=int, default=0,
                        help="Bounding-box shift (v1 only)")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output FPS (used when ref is image/directory)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="UNet inference batch size")
    parser.add_argument("--extra_margin", type=int, default=10,
                        help="Extra pixels below face bbox (v15 only)")
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--parsing_mode", type=str, default="jaw",
                        choices=["jaw", "neck", "raw"],
                        help="BiSeNet face-blending mode")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--use_saved_coord", action="store_true",
                        help="Load landmark cache from --coord_cache")
    parser.add_argument("--save_coord", action="store_true",
                        help="Save computed landmarks to --coord_cache")
    parser.add_argument("--coord_cache", type=str, default=None,
                        help="Path to landmark cache file (.pkl)")
    args = parser.parse_args()

    if not _check_ffmpeg():
        print("Warning: ffmpeg not found in PATH", file=sys.stderr)

    generate(args)
