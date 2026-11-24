# 🧠 MediScan AI - Brain MRI Analysis with Evidence Stability Verification

A beginner-friendly Streamlit web app that

1. classifies a brain MRI (glioma / meningioma / pituitary tumor / no tumor) and explains the prediction with a **Grad-CAM heatmap**, and
2. checks whether that evidence is **stable** when the image is changed in small, harmless ways, producing an **Evidence Stability Score (0-100)**.

> ⚠️ **Academic demonstration and research prototype only. Not a medical device, not for diagnosis.**
> Every output requires review by a qualified medical professional.
> The stability score measures **model consistency, not medical correctness.**

---

## 1. Project structure

```
mediscan_ai/
├── app.py            Streamlit interface: 4 pages + sidebar navigation
├── preprocessing.py  Image validation + resizing (shared by training and the app)
├── model.py          CNN definition, model loading, prediction
├── gradcam.py        Grad-CAM heatmap + overlay drawing
├── stability.py      Feature 2: image variations, comparison, Stability Score
├── charts.py         Matplotlib charts (probability bars, gauge, similarity bars)
├── report.py         PDF and HTML report generation
├── train.py          Training script (produces models/brain_mri_model.keras)
├── train_colab.ipynb Colab notebook that downloads the dataset and runs train.py
├── requirements.txt
├── .streamlit/config.toml   Teal/white theme
└── models/
    ├── brain_mri_model.keras   trained model (created by train.py)
    └── model_info.json         class names + measured test accuracy
```

## 2. Install and run

Requires **Python 3.10 - 3.12**.

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## 3. Dataset and model setup

The app needs a **trained model** in `models/`. The model is **not shipped in the zip**; train it once with
**Option 0 (Colab, easiest)** or the local steps below.

### Option 0 - Train on Google Colab (free GPU, ~10-15 min)

Open `train_colab.ipynb` in <https://colab.research.google.com>, set the runtime to **T4 GPU**, run all cells, and
unzip the downloaded `models.zip` into the project folder. Then run `streamlit run app.py`.

### Or train locally

### Download a dataset (choose ONE)

**Option A - Kaggle "Brain Tumor MRI Dataset"** (Masoud Nickparvar; ~7,000 images, 4 classes, larger and better):
<https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
Download, unzip, and rename the folder to `dataset` so you have `dataset/Training/...` and `dataset/Testing/...`
(class folders: `glioma`, `meningioma`, `notumor`, `pituitary`).

**Option B - GitHub "brain-tumor-classification-dataset"** (Sartaj Bhuvaji; ~3,260 images; no account needed):

```bash
git clone --depth 1 https://github.com/sartajbhuvaji/brain-tumor-classification-dataset.git dataset
```

(class folders: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`). `train.py` is written for this one (Option A also works).

### Train

```bash
python train.py --data_dir dataset --epochs 25
```

Training runs on a normal laptop CPU (roughly 1-2 minutes per epoch). When finished it prints the **test accuracy**
on images never used for training, and saves `models/brain_mri_model.keras` and `models/model_info.json`.
Restart the app afterwards.

### No model? Demo mode

If no trained model exists, the app does **not** pretend. It shows a red "No trained model found" notice with
instructions. A sidebar switch enables **DEMO mode**, which runs the pipeline with an **untrained** network.
Everything is then clearly labelled DEMO (including the report), and the results have no medical meaning.

## 4. How the two features work (plain English, for your presentation)

### Feature 1 - AI analysis + Grad-CAM

| Step | What happens | File |
|---|---|---|
| Upload & validate | Checks file type, size, that it opens as an image, minimum size, not blank; warns if it is a colour image | `preprocessing.py` |
| Preprocess | Convert to grayscale, resize to 128x128 | `preprocessing.py` |
| Predict | A small CNN outputs a probability for each of the 4 classes. The highest is the prediction, its probability is the confidence | `model.py` |
| Grad-CAM | Looks at the last convolution layer's feature maps, finds which ones pushed the predicted class score up (using gradients), adds them together with those weights, and paints the result as a heatmap: **red = most influence** | `gradcam.py` |

*Analogy:* the CNN is a student marking an X-ray; Grad-CAM is asking the student to circle the parts of the picture that made them decide.

### Feature 2 - Evidence Stability Verification

*Idea:* if tiny, harmless changes (a slightly brighter picture, a speck of noise) make the AI change its mind or look somewhere else, its evidence is fragile.

1. **8 variations** are created: brighter, darker, more contrast, less contrast, noise, rotated +5°, rotated −5°, resized 85%.
2. Each goes through the **same model**: new prediction, new confidence.
3. A **Grad-CAM heatmap for the original predicted class** is made for each (rotated heatmaps are rotated back so they can be compared fairly).
4. Three measurements:
   * **Prediction consistency** = % of variations with the same class as the original.
   * **Confidence stability** = 100 − average change in the original class's probability (in % points).
   * **Heatmap consistency** = average Pearson correlation between the original and each variation's heatmap (100 % = same highlighted areas).
5. **Evidence Stability Score = 40 % prediction + 20 % confidence + 40 % heatmap.**

| Score | Status | Meaning |
|---|---|---|
| 80 - 100 | **High stability** | Reliable-looking AI evidence |
| 60 - 79 | **Moderate stability** | Requires additional review |
| 0 - 59 | **Low stability** | Unstable AI evidence; doctor review recommended |

The weights and thresholds are constants at the top of `stability.py` and are easy to change. They are
reasonable design choices for a demonstration, **not clinically validated values**.

### Demo script (5 minutes)

1. **Home** - state the two features. Click *Start Analysis*.
2. **Image Analysis** - upload a test MRI (use one from `dataset/Testing`). Point out predicted class, confidence, and that the heatmap shows *where the model looked*.
3. **Stability Verification** - click *Run*. Show the gauge, the 8 variations, and the "Compare heatmaps" selector. Explain that a flipped class or a moved heatmap lowers the score.
4. **Analysis Report** - show the summary and download the PDF.
5. Finish with the limitation: *a high score means the model is consistent, not that it is right.*

## 5. Limitations (be upfront about these in your demo)

* The model is a small CNN trained on a public dataset. It always outputs one of the four classes, even for images that are not brain MRIs.
* Public Kaggle/GitHub brain-MRI datasets contain near-duplicate and augmented images, so accuracy numbers can be optimistic compared with real clinical data. No external clinical validation was done.
* Grad-CAM shows what influenced the *model*; it is not proof of where a tumor is.
* The stability variations are small, synthetic changes. They do not cover everything that differs between real scanners.
* Stability ≠ correctness: a model can be consistently wrong.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| "No trained model found" | Run `python train.py ...` (section 3) and restart the app |
| `ModuleNotFoundError` | Activate your virtual environment and run `pip install -r requirements.txt` |
| TensorFlow install fails | Use Python 3.10-3.12, or `pip install tensorflow-cpu` |
| Slow first run | TensorFlow loads once; later analyses are fast |
