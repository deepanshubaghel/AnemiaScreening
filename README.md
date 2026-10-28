# Pregnancy Anemia Screening

**Ready-to-run, two-checkpoint Streamlit app.** Both EfficientNet-B0 checkpoints include the original V5.1 model weights *plus* reconstructed CBC training-time preprocessing. No CSV upload, fold selection, or model selection is needed for end users.

## Install and run

Python 3.10–3.13 recommended (with a compatible PyTorch/TorchVision pair).

```bash
python -m pip install -r requirements.txt
python verify.py
python -m streamlit run app.py
```

Open the local Streamlit URL, normally `http://localhost:8501`.

### Patient workflow

Enter the optional patient/study ID, study centre, and **optional measured Hb (g/dL)**; enter all CBC values available directly from the lab report; upload pallor images (left/right conjunctiva, left/right fingernails, left/right palms, tongue); click **Run anemia screening**.

The app automatically routes **7/7** image views to the **complete-case** checkpoint and **1–6/7** to the **missing-aware** checkpoint. Zero images disables the button. Anemic/Non-Anemic is based on that checkpoint's binary sigmoid probability and own validation threshold (not `0.5`). A separate severity head returns Normal/Mild/Moderate/Severe. Hb is *never fed* to these checkpoints; its optional reference label is based on the study's `<11.0 g/dL` rule, shown separately, including disagreements.

### Exact data preparation

- Model weights are loaded with a strict `state_dict` match.
- All image inputs use the original training study's **eval transform**: `RGB → Resize((224,224)) → ToTensor() → Normalize(ImageNet mean/std)`.
- Missing image views receive the original zero (black) placeholder and `view_mask=0`, and are excluded by the attention mask.
- CBC feature names and their order are read from the selected checkpoint. The input is cast to `float32`; missing values are filled with the training median, then transformed using the fitted `StandardScaler` mean and scale as per the original V5.1 pipeline.
- The medians, means and scales are **embedded in each `.pt`**. No external patient-training CSV, original dataset, or scaler file is included or required by the application.

For this bundle, the model's training-time preprocessing was reconstructed from the user-provided `patient_split_and_fold_assignment.csv` according to `filter_for_experiment` and `prepare_tab_arrays` in `v5_1_MAViA_reduced.py`:

| Checkpoint | Eligible train samples (all `trainval` excluding fold 4) | Features | Threshold |
|---|---:|---:|---:|
| complete_case.pt | 281 (all seven views available) | 9 | 0.454122 |
| missing_aware.pt | 308 (one or more views available) | 9 | 0.583593 |

The original `.pt` files did not contain these fitted statistics. This deployment assumes the provided split CSV came from the **same experiment run** as the checkpoints; the CSV structure and recorded folds are compatible, but there is no cryptographic proof of its provenance in the checkpoint itself.

### Files

```
AnemiaScreening/
├── app.py                  # Patient-facing UI
├── model.py                # V5.1 neural network
├── router.py               # Automatic model selection
├── preprocessing.py        # Raw CBC conversion; no external files
├── verify.py               # Local inference smoke test
├── requirements.txt
├── .streamlit/config.toml
├── models/
│   ├── complete_case.pt    # Weights + exact fitted preprocessing
│   └── missing_aware.pt    # Weights + exact fitted preprocessing
└── build_models_once.py    # Optional developer-only regeneration utility
```

The private patient-level split CSV is **not distributed** with this application. The optional `build_models_once.py` is only useful when a developer replaces the checkpoints for a future training experiment and has that experiment's own split CSV; it is **never used during normal inference**.

**Research-use only:** Model predictions should not be treated as clinical diagnoses; independently validated hemoglobin measurements and clinical assessment remain essential. Model probabilities are not equivalent to independently calibrated clinical risks.


## Appearance: light, dark, and system theme

This version explicitly declares **both** `[theme.light]` and `[theme.dark]`
inside `.streamlit/config.toml`. This feature requires **Streamlit 1.51 or newer**.
The older deployment used a single `[theme]` configuration that could lock the
app to one theme and was compatible with Streamlit 1.50, which did not support
separate custom palettes.

1. Upgrade dependencies: `python -m pip install -U -r requirements.txt`.
2. **Stop and restart** the Streamlit server after extracting this version.
3. In the running browser app, open **⋮ → Settings → Theme** and choose
   **Light**, **Dark**, or **Use system setting**.
4. If the prior appearance remains cached, refresh the page. On a hosted
   Streamlit deployment, commit the updated `.streamlit/config.toml` and
   `requirements.txt` files, then reboot/redeploy the app.

The custom hero intentionally remains a readable navy-blue gradient in all
modes. All native input widgets, containers, captions, error messages, and
upload areas follow Streamlit's selected palette instead of fixed CSS colors.
All two-model routing logic and preprocessing files remain unchanged.
