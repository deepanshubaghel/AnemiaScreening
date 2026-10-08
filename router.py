"""Two-model routing. No historical training files are required at inference."""
from dataclasses import dataclass
from pathlib import Path
import torch
from model import MaviaV5Net
from preprocessing import CBCPreprocessor

MODEL_NAMES = {"complete": "complete_case.pt", "missing": "missing_aware.pt"}
EXPECTED_COLUMNS = [
    "cbc_Age", "cbc_TRBC", "cbc_PCV", "cbc_MCV", "cbc_MCH",
    "cbc_MCHC", "cbc_RDW", "cbc_Reticulocyte_count", "cbc_Reticulocyte_percent",
]


@dataclass
class Bundle:
    model: MaviaV5Net
    preprocessor: CBCPreprocessor
    threshold: float
    key: str
    checkpoint: dict


def select_model_key(image_count: int) -> str:
    if not 1 <= image_count <= 7:
        raise ValueError("Upload between one and seven images")
    return "complete" if image_count == 7 else "missing"


def load_bundle(base_dir: Path, key: str) -> Bundle:
    if key not in MODEL_NAMES:
        raise ValueError("Invalid model route")
    file = Path(base_dir) / "models" / MODEL_NAMES[key]
    if not file.is_file():
        raise FileNotFoundError(f"Bundled model file is missing: {file.name}")
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    cfg = checkpoint["exp_cfg"]
    expected_policy = "complete_selected" if key == "complete" else "missing_aware_selected"
    if cfg.get("sample_policy") != expected_policy:
        raise ValueError("Checkpoint's image policy does not match its route")
    if cfg.get("backbone") != "efficientnet_b0" or cfg.get("cbc_config") != "cbc_no_hb_full":
        raise ValueError("Incorrect model family or CBC configuration")
    if cfg.get("view_combo") != [
        "Left_Eye", "Right_Eye", "Left_Nail", "Right_Nail",
        "Left_Palm", "Right_Palm", "Tongue",
    ]:
        raise ValueError("Unexpected checkpoint image order")
    preproc = CBCPreprocessor.from_checkpoint(checkpoint)
    if list(preproc.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected checkpoint CBC feature order")
    network = MaviaV5Net(cfg, tab_dim=len(preproc.columns))
    network.load_state_dict(checkpoint["model_state"], strict=True)
    network.eval()
    return Bundle(
        model=network,
        preprocessor=preproc,
        threshold=float(checkpoint["best_threshold"]),
        key=key,
        checkpoint=checkpoint,
    )
