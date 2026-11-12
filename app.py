from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms as T

from router import load_bundle, select_model_key

APP_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 224
ANEMIA_HB_CUTOFF = 11.0
SEVERITY_NAMES = ["Normal", "Mild", "Moderate", "Severe"]

VIEW_INFO = [
    ("Left_Eye", "Left conjunctiva", "Conjunctiva"),
    ("Right_Eye", "Right conjunctiva", "Conjunctiva"),
    ("Left_Nail", "Left fingernail", "Fingernail"),
    ("Right_Nail", "Right fingernail", "Fingernail"),
    ("Left_Palm", "Left palm", "Palm"),
    ("Right_Palm", "Right palm", "Palm"),
    ("Tongue", "Tongue", "Dorsum of tongue"),
]

CBC_FIELDS = [
    ("cbc_TRBC", "RBC / TRBC", "10⁶/µL", "3.10", 0.01, 0.0, 10.0),
    ("cbc_PCV", "PCV / HCT", "%", "30.0", 0.1, 0.0, 80.0),
    ("cbc_MCV", "MCV", "fL", "88.0", 0.1, 0.0, 200.0),
    ("cbc_MCH", "MCH", "pg", "27.0", 0.1, 0.0, 80.0),
    ("cbc_MCHC", "MCHC", "g/dL", "31.5", 0.1, 0.0, 60.0),
    ("cbc_RDW", "RDW", "%", "15.5", 0.1, 0.0, 50.0),
    ("cbc_Reticulocyte_count", "Reticulocyte count", "10⁶/µL", "0.08", 0.001, 0.0, 5.0),
    ("cbc_Reticulocyte_percent", "Reticulocyte", "%", "1.8", 0.1, 0.0, 50.0),
]

IMAGE_TRANSFORM = T.Compose(
    [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

st.set_page_config(
    page_title="Pregnancy Anemia Screening",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.25rem;
    padding-bottom: 4rem;
}
.hero {
    background: linear-gradient(125deg, #153f65 0%, #246d9f 100%);
    color: white;
    border-radius: 22px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 12px 34px rgba(0,0,0,.12);
}
.hero h1 {
    color: white !important;
    margin: 0;
    font-size: 2.05rem;
    line-height: 1.2;
}
.hero p {
    color: rgba(255,255,255,.94) !important;
    margin: .7rem 0 0;
    line-height: 1.55;
    max-width: 960px;
}
.eyebrow {
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .11em;
    text-transform: uppercase;
}
.eyebrow { opacity: .85; margin-bottom: .55rem; }
.helper { opacity: .90; font-size: .91rem; line-height: 1.5; color: inherit; }
.route-pill {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(127,127,127,.30);
    border-radius: 999px;
    padding: .38rem .76rem;
    background: rgba(127,127,127,.07);
    color: inherit;
    font-size: .84rem;
    font-weight: 650;
}
.result-card {
    border: 1px solid rgba(127,127,127,.30);
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    background: rgba(127,127,127,.045);
    color: inherit;
    min-height: 142px;
}
.result-card.anemic { border-color: rgba(190,62,68,.58); }
.result-card.non-anemic { border-color: rgba(37,133,88,.58); }
.result-label {
    font-size: .69rem;
    font-weight: 800;
    letter-spacing: .09em;
    text-transform: uppercase;
    opacity: .82;
    color: inherit;
}
.result-value {
    font-size: 1.72rem;
    font-weight: 800;
    line-height: 1.12;
    margin: .45rem 0 .25rem;
    color: inherit;
}
.result-sub { font-size: .90rem; opacity: .84; line-height: 1.45; color: inherit; }
.view-title { font-weight: 750; margin-bottom: .05rem; color: inherit; }
.view-site { font-size: .80rem; opacity: .78; margin-bottom: .35rem; color: inherit; }
[data-testid="stFileUploader"] section { padding: .55rem; }
[data-testid="stMetricValue"] { font-size: 1.35rem; }
[data-testid="stNumberInput"] input { font-variant-numeric: tabular-nums; }
hr { border-color: rgba(127,127,127,.25) !important; }
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero { padding: 1.35rem; }
    .hero h1 { font-size: 1.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def card(title: str, value: str, subtitle: str, tone: str = "") -> None:
    tone_class = f" {tone}" if tone in {"anemic", "non-anemic"} else ""
    st.markdown(
        f'<div class="result-card{tone_class}">'
        f'<div class="result-label">{escape(title)}</div>'
        f'<div class="result-value">{escape(value)}</div>'
        f'<div class="result-sub">{escape(subtitle)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def hb_reference(hb_value: float | None) -> str | None:
    if hb_value is None:
        return None
    return "Anemic" if hb_value < ANEMIA_HB_CUTOFF else "Non-Anemic"


def open_upload(upload, label: str) -> Image.Image:
    try:
        upload.seek(0)
        return Image.open(upload).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{label} is not a valid image") from exc


def prepare_images(uploads: Dict[str, object]):
    tensors = []
    masks = []
    for key, label, _ in VIEW_INFO:
        upload = uploads.get(key)
        if upload is None:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            masks.append(0.0)
        else:
            image = open_upload(upload, label)
            masks.append(1.0)
        tensors.append(IMAGE_TRANSFORM(image))

    return (
        torch.stack(tensors, dim=0).unsqueeze(0),
        torch.tensor([masks], dtype=torch.float32),
    )


@st.cache_resource(show_spinner=False)
def get_bundle(route: str):
    return load_bundle(APP_DIR, route)


st.markdown(
    """
<div class="hero">
  <h1>Pregnancy Anemia Screening</h1>
  <p>Enter routine CBC values from the laboratory report and upload the available pallor images.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Patient -----------------------------
with st.container(border=True):
    st.subheader("Patient details")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        patient_id = st.text_input("Patient / Study ID", placeholder="Optional")
    with c2:
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=None,
            step=1,
            placeholder="e.g. 28",
            key="patient_age",
        )
    with c3:
        hb = st.number_input(
            "Hemoglobin (Hb), g/dL",
            min_value=0.0,
            max_value=25.0,
            value=None,
            step=0.1,
            format="%.1f",
            placeholder="Optional, e.g. 9.2",
        )

    c4, c5, c6 = st.columns(3, gap="large")
    with c4:
        place = st.text_input("Place / Village / Town", placeholder="Enter place")
    with c5:
        district = st.text_input("District", placeholder="Enter district")
    with c6:
        state = st.text_input("State", placeholder="Enter state")

    c7, c8, c9 = st.columns(3, gap="large")
    with c7:
        center = st.text_input("Study center", placeholder="Enter hospital or study center")
    with c8:
        pregnancy_status = st.selectbox(
            "Pregnancy status",
            ["Not specified", "Pregnant", "Not pregnant"],
            key="pregnancy_status",
        )
    with c9:
        trimester = st.selectbox(
            "Trimester",
            ["First trimester", "Second trimester", "Third trimester", "Unknown"],
            index=None,
            placeholder=("Select trimester" if pregnancy_status == "Pregnant" else "Not applicable"),
            disabled=pregnancy_status != "Pregnant",
            key="patient_trimester",
        )
    st.caption(
        "Hemoglobin is optional and used only for an independent study-reference comparison. "
        "Location and pregnancy details are study information and do not affect the current model prediction."
    )

# ----------------------------- CBC -----------------------------
with st.container(border=True):
    st.subheader("CBC parameters")
    st.markdown(
        '<div class="helper">Enter normal laboratory values directly. Leave a field blank only when that CBC measurement is unavailable; missing values are handled internally using the selected checkpoint\'s training-time preprocessing.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # Age stays in the checkpoint's nine-feature CBC vector, even though it's entered in Patient details.
    raw_values: dict[str, float | None] = {"cbc_Age": float(age) if age is not None else None}
    cbc_columns = st.columns(3, gap="large")
    for index, (key, label, unit, example, step, minimum, maximum) in enumerate(CBC_FIELDS):
        with cbc_columns[index % 3]:
            raw_values[key] = st.number_input(
                f"{label} ({unit})",
                min_value=float(minimum),
                max_value=float(maximum),
                value=None,
                step=float(step),
                placeholder=f"e.g. {example}",
                key=f"cbc_{key}",
            )

# ----------------------------- Images -----------------------------
with st.container(border=True):
    st.subheader("Patient images")
    st.markdown("#### Pallor photographs")
    st.caption(
        "Upload every available view. Seven views use the complete-view checkpoint automatically; one to six views use the missing-view checkpoint automatically."
    )

    uploads: Dict[str, object] = {}
    image_columns = st.columns(4, gap="medium")
    for index, (key, label, site) in enumerate(VIEW_INFO):
        with image_columns[index % 4]:
            st.markdown(f'<div class="view-title">{escape(label)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="view-site">{escape(site)}</div>', unsafe_allow_html=True)
            upload = st.file_uploader(
                f"Upload {label}",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"image_{key}",
                label_visibility="collapsed",
            )
            uploads[key] = upload
            if upload is not None:
                try:
                    preview = open_upload(upload, label)
                    preview.thumbnail((380, 220))
                    st.image(preview, width="stretch")
                except Exception:
                    pass

    # ----------------------- PBS microscopy preview -----------------------
    # This independent, future-facing uploader has no connection to model inputs.
    # Keep it strictly outside the seven-view `uploads` dictionary and inference.
    st.divider()
    st.markdown("#### PBS microscopy · Coming soon")
    st.caption("Demo preview only · Microscopic image screening will be available in a future update.")
    st.markdown("Upload **2–5 images each** for **40×** and **100×** magnification to preview your images. This section is optional for today's screening.")

    pbs_cols = st.columns(2, gap="large")
    for col, magnification in zip(pbs_cols, ("40×", "100×")):
        with col:
            pbs_files = st.file_uploader(
                f"PBS microscopic images — {magnification}",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key=f"pbs_{magnification}",
                help="Upload between 2 and 5 images for this magnification; preview only.",
            )
            if len(pbs_files) > 5:
                st.warning(f"{magnification}: Maximum 5 images. Remove {len(pbs_files) - 5} extra image(s) to complete the preview set.")
            elif len(pbs_files) == 1:
                st.info(f"{magnification}: Add at least 1 more image for a complete preview set (2–5).")
            elif 2 <= len(pbs_files) <= 5:
                st.caption(f"{len(pbs_files)}/5 images selected · Preview ready")
            else:
                st.caption("2–5 images per magnification")

            # Preview no more than five files, arranged in compact thumbnails.
            if pbs_files:
                thumb_cols = st.columns(2, gap="small")
                for image_index, uploaded in enumerate(pbs_files[:5], start=1):
                    with thumb_cols[(image_index - 1) % 2]:
                        try:
                            preview = open_upload(uploaded, f"PBS {magnification} image {image_index}")
                            preview.thumbnail((270, 160))
                            st.image(preview, caption=f"{magnification} · Image {image_index}", width="stretch")
                        except ValueError:
                            st.warning(f"Could not preview {uploaded.name}; please choose a valid image.")

image_count = sum(value is not None for value in uploads.values())
lab_count = sum(raw_values[key] is not None for key, *_ in CBC_FIELDS)
cbc_count = lab_count + int(age is not None)

if image_count == 7:
    route_label = "Complete-view model · all 7 images"
elif image_count > 0:
    route_label = f"Missing-view model · {image_count}/7 images"
else:
    route_label = "Upload at least one image"

# ----------------------------- Review -----------------------------
with st.container(border=True):
    st.subheader("Ready for screening")
    m1, m2, m3 = st.columns(3, gap="large")
    m1.metric("Images available", f"{image_count}/7")
    m2.metric("Age + CBC values entered", f"{cbc_count}/9")
    m3.metric("Hemoglobin", "Optional" if hb is None else f"{hb:.1f} g/dL")
    st.markdown(
        f'<span class="route-pill">Automatic model selection: {escape(route_label)}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    if image_count == 0:
        st.info("Upload at least one pallor image to enable screening.")
    if lab_count == 0:
        st.info("Enter at least one laboratory CBC value before screening. Age alone is not sufficient.")
    elif cbc_count < 7:
        st.warning("Several CBC measurements are missing. Prediction reliability may be reduced; enter available values where possible.")

    run = st.button(
        "Run anemia screening",
        type="primary",
        width="stretch",
        disabled=image_count == 0 or lab_count == 0,
    )

# ----------------------------- Inference -----------------------------
if run:
    try:
        route = select_model_key(image_count)
        bundle = get_bundle(route)
        tabular = bundle.preprocessor.transform(raw_values)
        images, view_mask = prepare_images(uploads)

        if route == "complete" and not bool(torch.all(view_mask == 1)):
            raise RuntimeError("Complete-view model received a missing image")

        with st.spinner("Analyzing patient data…"):
            with torch.inference_mode():
                output = bundle.model(images, view_mask, tabular)
                anemia_probability = float(torch.sigmoid(output["binary_logit"])[0].item())
                severity_probabilities = torch.softmax(output["severity_logits"], dim=1)[0].cpu()
                attention = output.get("attention_weights")
                attention = attention[0].cpu() if attention is not None else None

        binary_result = (
            "Anemic" if anemia_probability >= bundle.threshold else "Non-Anemic"
        )
        severity_index = int(torch.argmax(severity_probabilities).item())
        severity_result = SEVERITY_NAMES[severity_index]
        hb_result = hb_reference(hb)

        st.markdown("---")
        st.subheader("Patient-level result")

        r1, r2, r3 = st.columns(3, gap="large")
        with r1:
            card(
                "AI anemia prediction",
                binary_result,
                f"Anemia probability {anemia_probability:.1%}",
                "anemic" if binary_result == "Anemic" else "non-anemic",
            )
        with r2:
            card(
                "AI severity",
                severity_result,
                "Normal · Mild · Moderate · Severe",
            )
        with r3:
            if hb_result is None:
                card(
                    "Hb reference",
                    "Not entered",
                    f"Optional · study cutoff {ANEMIA_HB_CUTOFF:.1f} g/dL",
                )
            else:
                card(
                    "Hb study reference",
                    hb_result,
                    f"Hb {hb:.1f} g/dL · cutoff {ANEMIA_HB_CUTOFF:.1f} g/dL",
                    "anemic" if hb_result == "Anemic" else "non-anemic",
                )

        st.progress(anemia_probability, text=f"Anemia probability: {anemia_probability:.1%}")

        if hb_result is not None:
            if hb_result == binary_result:
                st.success("AI screening agrees with the Hb-based study classification.")
            else:
                st.warning(
                    "AI screening and the Hb-based study classification differ for this case. Review the case before interpretation."
                )

        if (binary_result == "Anemic") != (severity_result != "Normal"):
            st.warning(
                f"The two model heads differ for this case: anemia result = {binary_result}; severity result = {severity_result}."
            )

        with st.expander("Prediction details", expanded=False):
            st.write(
                "**Model selected automatically:** "
                + ("Complete-view" if route == "complete" else "Missing-view")
            )
            st.write(f"**Images used:** {image_count}/7")
            st.write(f"**Decision threshold:** {bundle.threshold:.3f}")
            st.write(f"**Anemia probability:** {anemia_probability:.4f}")

            severity_frame = pd.DataFrame(
                {
                    "Severity": SEVERITY_NAMES,
                    "Probability": [float(x) for x in severity_probabilities.tolist()],
                }
            )
            st.dataframe(severity_frame, hide_index=True, width="stretch")

            if attention is not None:
                attention_frame = pd.DataFrame(
                    {
                        "View": [label for _, label, _ in VIEW_INFO],
                        "Available": [bool(x) for x in view_mask[0].tolist()],
                        "Attention": [float(x) for x in attention.tolist()],
                    }
                )
                st.dataframe(attention_frame, hide_index=True, width="stretch")

        st.caption("Research-use screening interface; not a substitute for clinical diagnosis. Hb reference and AI outputs are independent.")

    except ValueError as exc:
        st.error(f"Please review the entered data: {exc}")
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Prediction could not be completed. Please verify the entered values and image files.")
        with st.expander("Technical details"):
            st.code(str(exc))
