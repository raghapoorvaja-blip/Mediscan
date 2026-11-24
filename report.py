"""
report.py
---------
Builds the downloadable analysis report in two formats:
  * build_html_report()  -> one self-contained HTML file (images embedded)
  * build_pdf_report()   -> a PDF made with Matplotlib (no extra library needed)

Both take the same `data` dictionary created by the app (see app.py, page_report).
"""

import base64
import html
import io
import textwrap

import cv2
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

DISCLAIMER = (
    "MediScan AI is an academic demonstration and research prototype. It is NOT a medical "
    "device and must NOT be used for diagnosis or treatment decisions. All outputs require "
    "review by a qualified medical professional."
)
STABILITY_NOTE = (
    "The Evidence Stability Score measures how CONSISTENT the AI model is when the image is "
    "slightly changed. It does NOT measure whether the prediction is medically CORRECT."
)


def reliability_summary(data):
    """Write the plain-English 'final AI reliability summary' from the real results."""
    lines = []
    if data.get("demo"):
        lines.append("DEMO MODE: these results come from an UNTRAINED network and have no medical meaning.")
    lines.append(
        f"The model classified this scan as '{data['predicted_name']}' with "
        f"{data['confidence']:.1f}% confidence."
    )
    if data["confidence"] < 60:
        lines.append("This confidence is low, so the model is uncertain about this image.")

    stab = data.get("stability")
    if stab is None:
        lines.append("Evidence Stability Verification has NOT been run for this image, so the "
                     "reliability of the AI evidence is unknown.")
    else:
        count = len(stab["variations"])
        lines.append(
            f"Across {count} small controlled changes, the prediction stayed the same in "
            f"{stab['prediction_consistency']:.0f}% of cases and the heatmaps were "
            f"{stab['heatmap_consistency']:.0f}% consistent with the original."
        )
        lines.append(
            f"Overall Evidence Stability Score: {stab['score']:.0f}/100 - {stab['status']}. {stab['message']}"
        )
    lines.append(STABILITY_NOTE)
    return lines


# ----------------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------------
def _png_data_uri(rgb_image):
    """Encode an RGB array as a base64 PNG so it can be embedded inside the HTML file."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) if rgb_image.ndim == 3 else rgb_image
    _, buffer = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buffer).decode()


def build_html_report(data):
    e = html.escape
    stab = data.get("stability")
    colours = {"high": "#12A37F", "moderate": "#F2A93B", "low": "#E5534B"}

    demo_banner = ""
    if data.get("demo"):
        demo_banner = ('<div style="background:#FFF4E5;border:2px solid #F2A93B;padding:10px;'
                       'border-radius:8px;margin-bottom:14px"><b>DEMO MODE</b> - untrained network. '
                       'Results are not real AI analysis.</div>')

    stability_html = "<p><i>Evidence Stability Verification was not run for this image.</i></p>"
    if stab:
        rows = "".join(
            f"<tr><td>{e(v['name'])}</td><td>{e(v['description'])}</td><td>{e(v['predicted_name'])}</td>"
            f"<td>{v['confidence']:.1f}%</td><td>{'Yes' if v['same_prediction'] else 'NO'}</td>"
            f"<td>{v['heatmap_similarity']:.0f}%</td></tr>"
            for v in stab["variations"]
        )
        stability_html = f"""
        <div class="cards">
          <div class="card"><div class="lab">Prediction consistency</div><div class="val">{stab['prediction_consistency']:.0f}%</div></div>
          <div class="card"><div class="lab">Heatmap consistency</div><div class="val">{stab['heatmap_consistency']:.0f}%</div></div>
          <div class="card"><div class="lab">Confidence stability</div><div class="val">{stab['confidence_stability']:.0f}%</div></div>
          <div class="card" style="border-color:{colours[stab['status_key']]}"><div class="lab">Evidence Stability Score</div>
            <div class="val" style="color:{colours[stab['status_key']]}">{stab['score']:.0f}/100</div>
            <div>{e(stab['status'])}</div></div>
        </div>
        <table><tr><th>Variation</th><th>Change</th><th>Prediction</th><th>Confidence</th><th>Same as original?</th>
        <th>Heatmap similarity</th></tr>{rows}</table>"""

    summary_html = "".join(f"<p>{e(line)}</p>" for line in reliability_summary(data))

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>MediScan AI Report</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;color:#12343B;max-width:900px;margin:30px auto;padding:0 16px}}
 h1{{color:#0B6E78;border-bottom:4px solid #0E9AA7;padding-bottom:8px}} h2{{color:#0B6E78;margin-top:28px}}
 .imgs{{display:flex;gap:12px}} .imgs div{{flex:1;text-align:center;font-size:13px}} .imgs img{{width:100%;border-radius:8px}}
 .cards{{display:flex;gap:12px;margin:12px 0}} .card{{flex:1;border:2px solid #BFE6EA;border-radius:10px;padding:10px;text-align:center}}
 .lab{{font-size:12px;color:#557}} .val{{font-size:26px;font-weight:bold}}
 table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border:1px solid #BFE6EA;padding:6px;text-align:left}}
 th{{background:#E6F7F8}} .disc{{background:#E6F7F8;padding:10px;border-radius:8px;font-size:13px;margin-top:24px}}
</style></head><body>
<h1>MediScan AI - Analysis Report</h1>{demo_banner}
<p><b>File:</b> {e(data['filename'])} &nbsp; <b>Generated:</b> {e(data['timestamp'])}<br>
<b>Model:</b> {e(data['model_description'])}</p>
<h2>1. Prediction</h2>
<div class="cards">
  <div class="card"><div class="lab">Predicted class</div><div class="val" style="font-size:20px">{e(data['predicted_name'])}</div></div>
  <div class="card"><div class="lab">Confidence</div><div class="val">{data['confidence']:.1f}%</div></div>
</div>
<div class="imgs">
  <div><img src="{_png_data_uri(data['original_rgb'])}"><br>Original MRI</div>
  <div><img src="{_png_data_uri(data['heatmap_rgb'])}"><br>Grad-CAM heatmap</div>
  <div><img src="{_png_data_uri(data['overlay_rgb'])}"><br>Overlay</div>
</div>
<p style="font-size:13px">Red/yellow areas influenced the prediction most; blue areas influenced it least.</p>
<h2>2. Evidence Stability Verification</h2>{stability_html}
<h2>3. Final AI reliability summary</h2>{summary_html}
<div class="disc">{e(DISCLAIMER)}</div>
</body></html>""".encode("utf-8")


# ----------------------------------------------------------------------------
# PDF report (one A4 page drawn with Matplotlib)
# ----------------------------------------------------------------------------
def build_pdf_report(data):
    stab = data.get("stability")
    status_colour = {"high": "#12A37F", "moderate": "#F2A93B", "low": "#E5534B"}

    fig = Figure(figsize=(8.27, 11.69))               # A4 portrait
    fig.patch.set_facecolor("white")

    # Title band
    band = fig.add_axes([0, 0.94, 1, 0.06])
    band.axis("off")
    band.add_patch(Rectangle((0, 0), 1, 1, color="#0E9AA7"))
    band.text(0.04, 0.5, "MediScan AI - Analysis Report", color="white", fontsize=17,
              fontweight="bold", va="center")

    y = 0.925
    fig.text(0.05, y, f"File: {data['filename']}     Generated: {data['timestamp']}", fontsize=8.5, color="#12343B")
    y -= 0.016
    fig.text(0.05, y, f"Model: {data['model_description']}", fontsize=8.5, color="#12343B")
    if data.get("demo"):
        y -= 0.022
        fig.text(0.05, y, "DEMO MODE - untrained network. Results are NOT real AI analysis.",
                 fontsize=10, color="#C0392B", fontweight="bold")

    # Three images side by side
    titles = ["Original MRI", "Grad-CAM heatmap", "Overlay"]
    images = [data["original_rgb"], data["heatmap_rgb"], data["overlay_rgb"]]
    for i, (title, image) in enumerate(zip(titles, images)):
        ax = fig.add_axes([0.05 + i * 0.31, 0.60, 0.28, 0.20])
        ax.imshow(image if image.ndim == 3 else image, cmap="gray")
        ax.set_title(title, fontsize=9, color="#0B6E78")
        ax.axis("off")

    fig.text(0.05, 0.575, f"Predicted class: {data['predicted_name']}", fontsize=12,
             fontweight="bold", color="#0B6E78")
    fig.text(0.05, 0.555, f"Confidence: {data['confidence']:.1f}%", fontsize=11, color="#12343B")

    # Stability section
    y = 0.52
    fig.text(0.05, y, "Evidence Stability Verification", fontsize=12, fontweight="bold", color="#0B6E78")
    y -= 0.022
    if stab is None:
        fig.text(0.05, y, "Not run for this image.", fontsize=9.5, color="#12343B")
        y -= 0.02
    else:
        fig.text(0.05, y, f"Prediction consistency: {stab['prediction_consistency']:.0f}%     "
                          f"Heatmap consistency: {stab['heatmap_consistency']:.0f}%     "
                          f"Confidence stability: {stab['confidence_stability']:.0f}%",
                 fontsize=9.5, color="#12343B")
        y -= 0.026
        fig.text(0.05, y, f"Evidence Stability Score: {stab['score']:.0f}/100 - {stab['status']}",
                 fontsize=12, fontweight="bold", color=status_colour[stab["status_key"]])
        y -= 0.024
        header = f"{'Variation':<15}{'Prediction':<20}{'Conf.':>8}{'Same?':>8}{'Heatmap sim.':>15}"
        fig.text(0.05, y, header, fontsize=8.5, family="monospace", fontweight="bold", color="#12343B")
        for v in stab["variations"]:
            y -= 0.017
            row = (f"{v['name']:<15}{v['predicted_name']:<20}{v['confidence']:>7.1f}%"
                   f"{('Yes' if v['same_prediction'] else 'NO'):>8}{v['heatmap_similarity']:>14.0f}%")
            fig.text(0.05, y, row, fontsize=8.5, family="monospace", color="#12343B")
        y -= 0.01

    # Final summary
    y -= 0.03
    fig.text(0.05, y, "Final AI reliability summary", fontsize=12, fontweight="bold", color="#0B6E78")
    for line in reliability_summary(data):
        for wrapped in textwrap.wrap(line, 105):
            y -= 0.0165
            fig.text(0.05, y, wrapped, fontsize=8.8, color="#12343B")
        y -= 0.004

    # Disclaimer footer
    foot = fig.add_axes([0.04, 0.02, 0.92, 0.065])
    foot.axis("off")
    foot.add_patch(Rectangle((0, 0), 1, 1, color="#E6F7F8"))
    foot.text(0.02, 0.5, "\n".join(textwrap.wrap(DISCLAIMER, 120)), fontsize=8, va="center", color="#12343B")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf")
    return buffer.getvalue()
