"""
Standalone Flask microservice exposing CLIP + HPSv2 scoring.

Start independently before running the main pipeline:

  $env:PYTHONNOUSERSITE='1'
  python score_service.py

The main pipeline calls http://localhost:5001/score.
If this service is unreachable, the pipeline degrades gracefully.

Expected local weights:

    models/clip/
      config.json
      pytorch_model.bin
      tokenizer files...

    models/hpsv2/
      HPS_v2.1_compressed.pt
      open_clip_pytorch_model.bin
"""

import base64
import io
import os

from flask import Flask, jsonify, request
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLIP_MODEL_DIR = os.path.join(MODEL_DIR, "clip")
CLIP_MODEL_SOURCE = os.environ.get(
    "STRUCTSVG_CLIP_MODEL",
    CLIP_MODEL_DIR if os.path.isdir(CLIP_MODEL_DIR) else "openai/clip-vit-base-patch32",
)
HPS_MODEL_DIR = os.path.join(MODEL_DIR, "hpsv2")
HPS_V21_CKPT = os.path.join(HPS_MODEL_DIR, "HPS_v2.1_compressed.pt")
HPS_OPENCLIP_CKPT = os.path.join(HPS_MODEL_DIR, "open_clip_pytorch_model.bin")

# hpsv2 uses HPS_ROOT as its cache/download root. Set it before import so any
# package-managed files stay inside the project.
os.environ.setdefault("HPS_ROOT", HPS_MODEL_DIR)

try:
    import hpsv2 as _hpsv2

    try:
        from hpsv2 import img_score as _hps_img_score
    except Exception:
        _hps_img_score = None

    _HPS_AVAILABLE = True
except ImportError:
    print("[Warning] hpsv2 not installed; HPS scores will return 0.0")
    _hpsv2 = None
    _hps_img_score = None
    _HPS_AVAILABLE = False


print(f"Loading CLIP model from {CLIP_MODEL_SOURCE} ...")

_clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_SOURCE,
    local_files_only=os.path.isdir(CLIP_MODEL_SOURCE),
).eval()
_clip_proc = CLIPProcessor.from_pretrained(
    CLIP_MODEL_SOURCE,
    local_files_only=os.path.isdir(CLIP_MODEL_SOURCE),
)
print("CLIP ready.")

if _HPS_AVAILABLE:
    missing = [p for p in (HPS_V21_CKPT, HPS_OPENCLIP_CKPT) if not os.path.isfile(p)]
    if missing:
        print("[Warning] HPSv2 local weights missing; HPS scores will return 0.0:")
        for path in missing:
            print(f"  - {path}")
        _HPS_AVAILABLE = False
    else:
        print(f"HPSv2 local weights ready in {HPS_MODEL_DIR}.")


app = Flask(__name__)


def _decode_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _clip_score(img: Image.Image, prompt: str) -> float:
    """
    Raw CLIP logits_per_image are dot-product similarities. Divide by 100 to
    keep the existing project scoring convention in a rough [0, 1] range.
    """
    inputs = _clip_proc(
        text=[prompt],
        images=img,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = _clip_model(**inputs)
        raw = outputs.logits_per_image.item()
    return max(0.0, min(1.0, raw / 100.0))


def _hps_score(img: Image.Image, prompt: str) -> float:
    if not _HPS_AVAILABLE:
        return 0.0
    try:
        if _hps_img_score is not None:
            score = _hps_img_score.score(
                img,
                prompt,
                cp=HPS_V21_CKPT,
                hps_version="v2.1",
            )
        else:
            score = _hpsv2.score(img, prompt, hps_version="v2.1")
        return float(score[0]) if hasattr(score, "__len__") else float(score)
    except Exception as exc:
        print(f"[HPSv2] Scoring failed: {exc}")
        return 0.0


@app.route("/score", methods=["POST"])
def score():
    """
    Expects JSON body: { "image_b64": "<base64 PNG>", "prompt": "<text>" }
    Returns:          { "clip": float, "hps": float }
    """
    data = request.get_json(force=True)
    if not data or "image_b64" not in data or "prompt" not in data:
        return jsonify({"error": "missing image_b64 or prompt"}), 400

    try:
        img = _decode_image(data["image_b64"])
        prompt = str(data["prompt"])
        return jsonify({
            "clip": _clip_score(img, prompt),
            "hps": _hps_score(img, prompt),
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "clip": 0.0, "hps": 0.0}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "clip_model": CLIP_MODEL_SOURCE,
        "hps_available": _HPS_AVAILABLE,
        "hps_model_dir": HPS_MODEL_DIR,
    })


if __name__ == "__main__":
    host = os.environ.get("STRUCTSVG_SCORER_HOST", "127.0.0.1")
    port = int(os.environ.get("STRUCTSVG_SCORER_PORT", "5001"))
    print(f"Scorer service starting on http://{host}:{port} ...")
    app.run(host=host, port=port, threaded=False)
