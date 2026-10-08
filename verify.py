"""Self-contained smoke test: both embedded checkpoints, CBC transform and AI forward passes."""
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from router import load_bundle, select_model_key, EXPECTED_COLUMNS

BASE = Path(__file__).resolve().parent

def verify():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    assert select_model_key(7) == "complete"
    assert all(select_model_key(i) == "missing" for i in range(1, 7))
    try:
        select_model_key(0)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero uploaded images must be rejected")
    print("[OK] Automatic checkpoint routing")
    for key in ("complete", "missing"):
        bundle = load_bundle(BASE, key)
        raw = dict(zip(EXPECTED_COLUMNS,
                       [28.0,2.64,27.0,116.2,31.0,31.5,15.5,0.37,13.92]))
        cbc = bundle.preprocessor.transform(raw)
        assert cbc.shape == (1,9) and torch.isfinite(cbc).all()
        raw[EXPECTED_COLUMNS[5]] = None
        assert torch.isfinite(bundle.preprocessor.transform(raw)).all()
        prep = T.Compose([T.Resize((224,224)),T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        transformed = prep(Image.new("RGB",(230,230),(148,105,95)))
        images = transformed.unsqueeze(0).repeat(7,1,1,1).unsqueeze(0)
        mask = torch.ones(1,7)
        if key == "missing":
            mask[:,1:]=0
        with torch.inference_mode():
            result = bundle.model(images,mask,cbc)
            p = torch.sigmoid(result["binary_logit"])
            q = torch.softmax(result["severity_logits"],dim=1)
        assert p.shape==(1,) and 0<=p.item()<=1 and torch.isfinite(p).all()
        assert q.shape==(1,4) and torch.allclose(q.sum(dim=1),torch.ones(1),atol=1e-6)
        assert result["attention_weights"].shape == (1,7)
        assert torch.isclose(result["attention_weights"].sum(),torch.tensor(1.0),atol=1e-6)
        if key=='missing':
            assert (result["attention_weights"][0,1:]==0).all()
        print(f"[OK] {key}: strictly loaded, CBC medians/scales embedded, forward pass successful")
    print("VERIFICATION PASSED")

if __name__ == "__main__":
    verify()
