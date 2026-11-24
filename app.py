"""
app.py  -  MediScan AI (Streamlit front end)
============================================
Run with:   streamlit run app.py

Four pages, selected from the sidebar:
    Home  ->  Image Analysis  ->  Stability Verification  ->  Analysis Report

Everything the user uploads or computes is kept in st.session_state, so results
are still there when you move between pages.
"""

import hashlib
import html
import time

import numpy as np
import streamlit as st

from charts import probability_chart, score_gauge, similarity_chart
from gradcam import compute_gradcam, heatmap_to_rgb, overlay_heatmap
from model import build_demo_model, display_name, load_trained_model, predict_probs, MODEL_PATH
from preprocessing import ALLOWED_EXTENSIONS, preprocess_batch, validate_and_load
from report import DISCLAIMER, STABILITY_NOTE, build_html_report, build_pdf_report, reliability_summary
from stability import (HIGH_THRESHOLD, MODERATE_THRESHOLD, WEIGHT_CONFIDENCE, WEIGHT_HEATMAP,
                       WEIGHT_PREDICTION, run_stability_analysis)

st.set_page_config(page_title="MediScan AI", page_icon="🧠", layout="wide")

PAGES = ["Home", "Image Analysis", "Stability Verification", "Analysis Report"]
TEAL, TEAL_DARK = "#0E9AA7", "#0B6E78"
STATUS_COLOURS = {"high": "#12A37F", "moderate": "#E59A1F", "low": "#E5534B"}

# ----------------------------------------------------------------------------
# Visual style (white + teal/cyan)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  section[data-testid="stSidebar"] {background: #F1FAFB; border-right: 1px solid #CDEBEE;}
  h1, h2, h3 {color: #0B6E78;}
  [data-testid="stImage"] img {border-radius: 10px; border: 1px solid #CDEBEE;}
  .hero {background: linear-gradient(135deg, #0E9AA7 0%, #3CC7D3 100%); color: white;
         padding: 34px 38px; border-radius: 18px; margin-bottom: 22px;}
  .hero h1 {color: white; margin: 0 0 8px 0; font-size: 2.4rem;}
  .hero p {font-size: 1.05rem; margin: 0; opacity: 0.95;}
  .card {background: white; border: 1px solid #CDEBEE; border-radius: 14px; padding: 16px 18px;
         box-shadow: 0 2px 8px rgba(14,154,167,0.08); height: 100%;}
  .card-label {font-size: 0.8rem; color: #5B7B82; text-transform: uppercase; letter-spacing: 0.04em;}
  .card-value {font-size: 1.8rem; font-weight: 700; line-height: 1.25;}
  .card-sub {font-size: 0.82rem; color: #5B7B82;}
  .banner {border-radius: 12px; padding: 14px 18px; margin: 8px 0 14px 0; border-left: 8px solid;}
  .note {background: #E6F7F8; border-radius: 10px; padding: 12px 16px; font-size: 0.9rem; color: #12343B;}
  .small {font-size: 0.85rem; color: #12343B;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Session state (the app's "memory" while you move between pages)
# ----------------------------------------------------------------------------
DEFAULT_STATE = {
    "nav": "Home",          # current page
    "demo_mode": False,     # demo mode switch (only offered when no trained model exists)
    "run_key": None,        # identifies the uploaded image the stored results belong to
    "filename": None,
    "gray": None,           # the uploaded image as a grayscale array
    "warnings": [],
    "analysis": None,       # Feature 1 results
    "stability": None,      # Feature 2 results
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def go_to(page):
    """Button callback that switches the sidebar navigation to another page."""
    st.session_state.nav = page


def clear_analysis():
    for key in ("run_key", "filename", "gray", "warnings", "analysis", "stability"):
        st.session_state[key] = DEFAULT_STATE[key]


# ----------------------------------------------------------------------------
# Model loading (cached, so the model is loaded only once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model ...")
def get_trained_model():
    return load_trained_model()


@st.cache_resource(show_spinner="Preparing demo network ...")
def get_demo_model():
    return build_demo_model()


def get_active_model():
    """
    Decide which model to use.
      * trained model available      -> use it (real analysis)
      * not available + demo mode on -> untrained demo network, clearly labelled
      * otherwise                    -> None (the pages show setup instructions)
    """
    model, class_names, info, message = get_trained_model()
    if model is not None:
        accuracy = info.get("test_accuracy")
        description = "MediScan CNN (trained)"
        if accuracy is not None:
            description += f" - test accuracy {accuracy * 100:.1f}%"
        return {"model": model, "class_names": class_names, "info": info,
                "demo": False, "description": description, "message": message}
    if st.session_state.demo_mode:
        demo_model, demo_classes = get_demo_model(), ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        return {"model": demo_model, "class_names": demo_classes, "info": {}, "demo": True,
                "description": "UNTRAINED demo network (no medical meaning)", "message": message}
    return {"model": None, "message": message, "demo": False}


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def card(label, value, sub="", colour=TEAL_DARK):
    st.markdown(
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value" style="color:{colour}">{html.escape(str(value))}</div>'
        f'<div class="card-sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def banner(title, text, colour):
    st.markdown(
        f'<div class="banner" style="background:{colour}18;border-color:{colour}">'
        f'<b style="color:{colour}">{html.escape(title)}</b><br>'
        f'<span class="small">{html.escape(text)}</span></div>',
        unsafe_allow_html=True,
    )


def disclaimer():
    st.markdown(f'<div class="note"><b>Important:</b> {html.escape(DISCLAIMER)}</div>', unsafe_allow_html=True)


def demo_warning():
    st.warning("**DEMO MODE** - no trained model was found, so this uses an UNTRAINED network. "
               "Its predictions, heatmaps and scores are NOT real AI analysis and have no medical meaning. "
               "It only demonstrates how the interface and the stability check work.")


def show_missing_model_help(message):
    st.error("No trained model is available, so real AI analysis cannot be performed.")
    st.write(message)
    st.markdown("**To fix this** (see README.md for details):")
    st.code("python train.py --data_dir dataset --epochs 25", language="bash")
    st.write(f"This creates `{MODEL_PATH}`. Restart the app afterwards. "
             "Alternatively, switch on **Demo mode** in the sidebar to explore the interface "
             "with a clearly labelled untrained network.")


# ----------------------------------------------------------------------------
# FEATURE 1 back end: predict + Grad-CAM for the uploaded image
# ----------------------------------------------------------------------------
def run_analysis(gray, active):
    labels = [display_name(c) for c in active["class_names"]]
    batch = preprocess_batch([gray])                       # resize to what the model expects
    probs = predict_probs(active["model"], batch)[0]       # real model prediction
    predicted = int(np.argmax(probs))
    heatmap, _ = compute_gradcam(active["model"], batch, class_index=predicted)   # real Grad-CAM
    return {
        "labels": labels,
        "probs": probs,
        "predicted": predicted,
        "confidence": float(probs[predicted]) * 100,
        "heatmap": heatmap,
        "heatmap_rgb": heatmap_to_rgb(heatmap, size=gray.shape),
        "overlay": overlay_heatmap(gray, heatmap),
        "demo": active["demo"],
        "model_description": active["description"],
        "info": active["info"],
    }


# ----------------------------------------------------------------------------
# Page 1: Home
# ----------------------------------------------------------------------------
def page_home():
    st.markdown("""
    <div class="hero">
      <h1>🧠 MediScan AI</h1>
      <p>Brain MRI analysis with explainable AI - and a check that the AI's evidence stays stable.</p>
    </div>""", unsafe_allow_html=True)

    st.write(
        "MediScan AI analyses a brain MRI image, predicts a possible abnormality, shows "
        "**why** it predicted that, and then tests whether the prediction and explanation "
        "stay the same when the image is changed in small, harmless ways. "
        "The goal is not only to predict, but to check how **stable** the AI's evidence is."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class="card"><h3 style="margin-top:0">① AI Analysis + Explainable AI</h3>
        <ul class="small">
          <li>Upload a brain MRI image (it is validated first)</li>
          <li>A trained CNN predicts: glioma, meningioma, pituitary tumor, or no tumor</li>
          <li>Shows the confidence as a percentage</li>
          <li><b>Grad-CAM heatmap</b> highlights the regions that influenced the prediction</li>
          <li>Original image and heatmap are shown side by side</li>
        </ul></div>""", unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="card"><h3 style="margin-top:0">② Evidence Stability Verification</h3>
        <ul class="small">
          <li>Creates 8 small variations: brightness, contrast, noise, rotation, resizing</li>
          <li>Runs every variation through the same model, with its own Grad-CAM</li>
          <li>Compares predictions, confidence and heatmaps with the original</li>
          <li>Produces an <b>Evidence Stability Score (0-100)</b> and a reliability status</li>
        </ul></div>""", unsafe_allow_html=True)

    st.write("")
    st.button("Start Analysis  →", type="primary", on_click=go_to, args=("Image Analysis",))
    st.write("")
    disclaimer()


# ----------------------------------------------------------------------------
# Page 2: Image Analysis (Feature 1)
# ----------------------------------------------------------------------------
def page_analysis():
    st.title("🔬 Image Analysis")
    active = get_active_model()
    if active["model"] is None:
        show_missing_model_help(active["message"])
        return
    if active["demo"]:
        demo_warning()

    uploaded = st.file_uploader(
        "Upload a brain MRI image",
        type=[ext.strip(".") for ext in ALLOWED_EXTENSIONS],
        help="PNG, JPG, BMP, TIFF or WEBP, up to 10 MB.",
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        run_key = hashlib.md5(data).hexdigest() + ("-demo" if active["demo"] else "")
        if run_key != st.session_state.run_key:               # a new image (or a changed mode)
            gray, warnings, error = validate_and_load(data, uploaded.name)
            if error:
                clear_analysis()
                st.error(f"Invalid image: {error}")
                return
            with st.spinner("Analysing image ..."):
                analysis = run_analysis(gray, active)
            st.session_state.update(run_key=run_key, filename=uploaded.name, gray=gray,
                                    warnings=warnings, analysis=analysis, stability=None)

    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Upload an MRI image to begin.")
        return

    gray = st.session_state.gray
    for text in st.session_state.warnings:
        st.warning(text)
    st.caption(f"Current image: **{st.session_state.filename}**")

    # --- Prediction cards ------------------------------------------------------
    confidence = analysis["confidence"]
    conf_colour = STATUS_COLOURS["high"] if confidence >= 80 else STATUS_COLOURS["moderate"] if confidence >= 60 else STATUS_COLOURS["low"]
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Predicted class", analysis["labels"][analysis["predicted"]])
    with c2:
        card("Confidence", f"{confidence:.1f}%", "Model's probability for this class", conf_colour)
    with c3:
        accuracy = analysis["info"].get("test_accuracy")
        if analysis["demo"]:
            card("Model", "DEMO", "Untrained - not real AI", STATUS_COLOURS["low"])
        else:
            card("Model test accuracy", f"{accuracy * 100:.1f}%" if accuracy else "n/a",
                 f"on {analysis['info'].get('test_images', '?')} unseen images")
    if confidence < 60:
        st.warning("The model's confidence is low for this image - treat the prediction with extra caution.")

    # --- Original vs heatmap -----------------------------------------------------
    st.subheader("Original MRI and Grad-CAM explanation")
    i1, i2, i3 = st.columns(3)
    i1.image(gray, caption="Original MRI", width="stretch")
    i2.image(analysis["heatmap_rgb"], caption="Grad-CAM heatmap", width="stretch")
    i3.image(analysis["overlay"], caption="Heatmap overlaid on MRI", width="stretch")
    st.markdown(
        '<div class="note"><b>How to read the heatmap:</b> red/yellow areas influenced the prediction most, '
        "blue areas influenced it least. It shows where the <i>model</i> looked - "
        "it does not prove that these areas are a tumor.</div>", unsafe_allow_html=True)

    st.subheader("Probability for each class")
    st.pyplot(probability_chart(analysis["labels"], analysis["probs"], analysis["predicted"]), width="stretch")

    st.button("Continue to Stability Verification  →", type="primary", on_click=go_to,
              args=("Stability Verification",))
    st.write("")
    disclaimer()


# ----------------------------------------------------------------------------
# Page 3: Evidence Stability Verification (Feature 2)
# ----------------------------------------------------------------------------
def page_stability():
    st.title("🛡️ Evidence Stability Verification")
    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Please analyse an MRI image first.")
        st.button("Go to Image Analysis", on_click=go_to, args=("Image Analysis",))
        return

    active = get_active_model()
    if active["model"] is None:
        show_missing_model_help(active["message"])
        return
    if analysis["demo"]:
        demo_warning()

    st.write(
        "A reliable AI should give the same answer - and look at the same image regions - when the scan "
        "is changed in tiny, harmless ways. Here the image is slightly modified 8 different ways and "
        "everything is compared with the original result."
    )
    st.markdown(f'<div class="note"><b>Original prediction:</b> {html.escape(analysis["labels"][analysis["predicted"]])} '
                f'({analysis["confidence"]:.1f}% confidence)</div>', unsafe_allow_html=True)
    st.write("")

    label = "Re-run verification" if st.session_state.stability else "Run Evidence Stability Verification"
    if st.button(label, type="primary"):
        with st.spinner("Creating variations, predicting and generating heatmaps ..."):
            st.session_state.stability = run_stability_analysis(
                model=active["model"],
                class_names=analysis["labels"],                 # readable names
                gray=st.session_state.gray,
                original_class=analysis["predicted"],
                original_probs=analysis["probs"],
                original_heatmap=analysis["heatmap"],
            )

    stab = st.session_state.stability
    if stab is None:
        st.info("Press the button above to start. It takes a few seconds.")
        return

    # --- Score + status ------------------------------------------------------------
    colour = STATUS_COLOURS[stab["status_key"]]
    st.subheader("Evidence Stability Score")
    left, right = st.columns([1, 1.6])
    with left:
        st.pyplot(score_gauge(stab["score"], stab["status_key"]), width="stretch")
    with right:
        banner(stab["status"], stab["message"], colour)
        m1, m2, m3 = st.columns(3)
        with m1:
            card("Prediction consistency", f"{stab['prediction_consistency']:.0f}%",
                 "variations with the same class")
        with m2:
            card("Heatmap consistency", f"{stab['heatmap_consistency']:.0f}%", "similarity to original heatmap")
        with m3:
            card("Confidence stability", f"{stab['confidence_stability']:.0f}%", "steadiness of the probability")
    st.markdown(
        f'<div class="note"><b>Thresholds:</b> {HIGH_THRESHOLD}-100 = High stability (reliable-looking evidence) &nbsp;|&nbsp; '
        f'{MODERATE_THRESHOLD}-{HIGH_THRESHOLD - 1} = Moderate (requires additional review) &nbsp;|&nbsp; '
        f'below {MODERATE_THRESHOLD} = Low (unstable evidence, doctor review recommended).<br>'
        f'<b>{html.escape(STABILITY_NOTE)}</b></div>', unsafe_allow_html=True)
    with st.expander("How is the score calculated?"):
        st.write(
            f"**Score = {WEIGHT_PREDICTION:.0%} x Prediction consistency + {WEIGHT_CONFIDENCE:.0%} x Confidence stability "
            f"+ {WEIGHT_HEATMAP:.0%} x Heatmap consistency**\n\n"
            "- *Prediction consistency*: the % of modified images that get the same class as the original.\n"
            "- *Confidence stability*: 100 minus the average change (in percentage points) of the probability "
            "of the original class.\n"
            "- *Heatmap consistency*: the average correlation between the original Grad-CAM heatmap and the heatmap "
            "of each modified image (rotated images are rotated back first). 100% means identical highlighted areas."
        )

    # --- Variation gallery ----------------------------------------------------------
    st.subheader("Image variations and their predictions")
    variations = stab["variations"]
    for start in range(0, len(variations), 4):
        columns = st.columns(4)
        for column, v in zip(columns, variations[start:start + 4]):
            with column:
                st.image(v["image"], caption=f"{v['name']} ({v['description']})", width="stretch")
                st.image(v["overlay"], caption="Grad-CAM for original class", width="stretch")
                tick, tick_colour = ("✔ Same class", STATUS_COLOURS["high"]) if v["same_prediction"] else ("✘ Class changed", STATUS_COLOURS["low"])
                st.markdown(
                    f'<div class="small"><b style="color:{tick_colour}">{tick}</b><br>'
                    f'{html.escape(v["predicted_name"])} - {v["confidence"]:.1f}%<br>'
                    f'Heatmap similarity: <b>{v["heatmap_similarity"]:.0f}%</b></div>', unsafe_allow_html=True)

    # --- Table + chart ----------------------------------------------------------------
    st.subheader("Summary table")
    st.dataframe(
        [{"Variation": v["name"], "What changed": v["description"], "Prediction": v["predicted_name"],
          "Confidence (%)": round(v["confidence"], 1),
          "Same as original?": "Yes" if v["same_prediction"] else "NO",
          "Heatmap similarity (%)": round(v["heatmap_similarity"])} for v in variations],
        hide_index=True, width="stretch")

    st.subheader("Heatmap consistency per variation")
    st.pyplot(similarity_chart([v["name"] for v in variations], [v["heatmap_similarity"] for v in variations]),
              width="stretch")

    # --- Side-by-side heatmap comparison ------------------------------------------------
    st.subheader("Compare heatmaps")
    choice = st.selectbox("Choose a variation to compare with the original", [v["name"] for v in variations])
    chosen = next(v for v in variations if v["name"] == choice)
    gray = st.session_state.gray
    difference = np.abs(analysis["heatmap"] - chosen["heatmap"])
    d1, d2, d3 = st.columns(3)
    d1.image(analysis["overlay"], caption="Original heatmap", width="stretch")
    d2.image(overlay_heatmap(gray, chosen["heatmap"]), caption=f"{choice} heatmap (aligned to original)", width="stretch")
    d3.image(heatmap_to_rgb(difference, size=gray.shape), caption="Difference (red = big change)", width="stretch")

    st.button("Continue to Analysis Report  →", type="primary", on_click=go_to, args=("Analysis Report",))
    st.write("")
    disclaimer()


# ----------------------------------------------------------------------------
# Page 4: Analysis Report
# ----------------------------------------------------------------------------
def build_report_data():
    """Collect everything stored in session_state into one dictionary for report.py."""
    analysis, gray = st.session_state.analysis, st.session_state.gray
    return {
        "filename": st.session_state.filename,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "demo": analysis["demo"],
        "model_description": analysis["model_description"],
        "predicted_name": analysis["labels"][analysis["predicted"]],
        "confidence": analysis["confidence"],
        "original_rgb": np.stack([gray] * 3, axis=-1),
        "heatmap_rgb": analysis["heatmap_rgb"],
        "overlay_rgb": analysis["overlay"],
        "stability": st.session_state.stability,
    }


def page_report():
    st.title("📄 Analysis Report")
    if st.session_state.analysis is None:
        st.info("Please analyse an MRI image first.")
        st.button("Go to Image Analysis", on_click=go_to, args=("Image Analysis",))
        return

    data = build_report_data()
    stab = data["stability"]
    if data["demo"]:
        demo_warning()
    st.caption(f"File: **{data['filename']}**   |   Generated: {data['timestamp']}   |   Model: {data['model_description']}")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Predicted class", data["predicted_name"])
    with c2:
        card("Confidence", f"{data['confidence']:.1f}%")
    with c3:
        if stab:
            card("Evidence Stability", f"{stab['score']:.0f}/100", stab["status"], STATUS_COLOURS[stab["status_key"]])
        else:
            card("Evidence Stability", "Not run", "Run it on the previous page", STATUS_COLOURS["moderate"])

    st.subheader("Images")
    i1, i2, i3 = st.columns(3)
    i1.image(data["original_rgb"], caption="Original MRI", width="stretch")
    i2.image(data["heatmap_rgb"], caption="Grad-CAM heatmap", width="stretch")
    i3.image(data["overlay_rgb"], caption="Overlay", width="stretch")

    st.subheader("Stability results")
    if stab:
        s1, s2, s3 = st.columns(3)
        s1.metric("Prediction consistency", f"{stab['prediction_consistency']:.0f}%")
        s2.metric("Heatmap consistency", f"{stab['heatmap_consistency']:.0f}%")
        s3.metric("Confidence stability", f"{stab['confidence_stability']:.0f}%")
    else:
        st.info("Evidence Stability Verification has not been run for this image yet.")
        st.button("Run it now", on_click=go_to, args=("Stability Verification",))

    st.subheader("Final AI reliability summary")
    for line in reliability_summary(data):
        st.write("• " + line)

    st.subheader("Download")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇ Download PDF report", build_pdf_report(data), file_name="mediscan_report.pdf",
                       mime="application/pdf", type="primary")
    d2.download_button("⬇ Download HTML report", build_html_report(data), file_name="mediscan_report.html",
                       mime="text/html")
    st.write("")
    disclaimer()


# ----------------------------------------------------------------------------
# Sidebar + page routing
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:{TEAL_DARK};margin-bottom:0'>🧠 MediScan AI</h2>"
                "<div class='small'>Medical image analysis &amp; evidence stability</div>", unsafe_allow_html=True)
    st.write("")
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.divider()

    status = get_trained_model()
    if status[0] is not None:
        st.success("Trained model loaded")
    else:
        st.error("No trained model found")
        st.toggle("Enable DEMO mode (untrained network)", key="demo_mode")
        if st.session_state.demo_mode:
            st.warning("Demo mode is ON - results are not real.")
    st.caption("Academic demonstration only. Not for clinical use.")

{"Home": page_home, "Image Analysis": page_analysis,
 "Stability Verification": page_stability, "Analysis Report": page_report}[st.session_state.nav]()
