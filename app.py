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
    ("cbc_Age", "Age", "years", "28", 1.0, 0.0, 120.0),
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
.eyebrow, .section-kicker {
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .11em;
    text-transform: uppercase;
}
.eyebrow { opacity: .85; margin-bottom: .55rem; }
.section-kicker { opacity: .80; margin-bottom: .25rem; color: inherit; }
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

# Streamlit owns the native app palette. Its Settings menu exposes Light, Dark,
# and Use system setting because both [theme.light] and [theme.dark] exist.
# Never force native widget foreground/background colors using CSS: those must
# continue to react to the visitor's active Streamlit theme.

# st.caption("Appearance: ⋮ menu → Settings → Theme → Light, Dark, or Use system setting.")

# ----------------------------- Patient -----------------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">01 · Patient information</div>', unsafe_allow_html=True)
    st.subheader("Patient and hemoglobin")
    c1, c2, c3 = st.columns([1.15, 1.0, 1.0], gap="large")
    with c1:
        patient_id = st.text_input("Patient / Study ID", placeholder="Optional")
    with c2:
        center = st.selectbox(
            "Study center",
            ["Not specified", "Dhanbad", "Kolkata", "Rims", "Sadar"],
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
    st.caption(
        "Hb is optional and is shown only as an independent study-reference classification. It is not passed into these no-Hb neural-network checkpoints."
    )

# ----------------------------- CBC -----------------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">02 · Complete blood count</div>', unsafe_allow_html=True)
    st.subheader("CBC parameters")
    st.markdown(
        '<div class="helper">Enter normal laboratory values directly. Leave a field blank only when that CBC measurement is unavailable; missing values are handled internally using the selected checkpoint\'s training-time preprocessing.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    raw_values: dict[str, float | None] = {}
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
    st.markdown('<div class="section-kicker">03 · Pallor images</div>', unsafe_allow_html=True)
    st.subheader("Patient images")
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

image_count = sum(value is not None for value in uploads.values())
cbc_count = sum(value is not None for value in raw_values.values())

if image_count == 7:
    route_label = "Complete-view model · all 7 images"
elif image_count > 0:
    route_label = f"Missing-view model · {image_count}/7 images"
else:
    route_label = "Upload at least one image"

# ----------------------------- Review -----------------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">04 · Review and screen</div>', unsafe_allow_html=True)
    st.subheader("Ready for screening")
    m1, m2, m3 = st.columns(3, gap="large")
    m1.metric("Images available", f"{image_count}/7")
    m2.metric("CBC values entered", f"{cbc_count}/9")
    m3.metric("Hemoglobin", "Optional" if hb is None else f"{hb:.1f} g/dL")
    st.markdown(
        f'<span class="route-pill">Automatic model selection: {escape(route_label)}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    if image_count == 0:
        st.info("Upload at least one pallor image to enable screening.")
    if cbc_count == 0:
        st.info("Enter at least one CBC measurement before screening. All available CBC values are recommended.")
    elif cbc_count < 7:
        st.warning("Several CBC measurements are missing. Prediction reliability may be reduced; enter available values where possible.")

    run = st.button(
        "Run anemia screening",
        type="primary",
        width="stretch",
        disabled=image_count == 0 or cbc_count == 0,
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
        st.markdown('<div class="section-kicker">Screening result</div>', unsafe_allow_html=True)
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
