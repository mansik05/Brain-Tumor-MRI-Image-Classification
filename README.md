# Brain Tumor MRI Image Classification

An educational deep-learning project that classifies brain MRI images into four dataset classes: `glioma`, `meningioma`, `no_tumor`, and `pituitary`.

> **Safety notice:** This is an educational demonstration, not a medical device. Predictions are not diagnoses and must not be used for treatment or triage. Consult a qualified medical professional.

## Project features

- Image loading, resizing to 224 × 224, and preprocessing
- Data augmentation during training
- Custom Convolutional Neural Network (CNN) baseline
- MobileNetV2 transfer learning with fine-tuning
- Evaluation using accuracy, macro precision, macro recall, and macro F1-score
- Per-class classification reports, confusion matrices, and training-history plots
- CSV comparison of the trained models
- Streamlit app for uploading an MRI image and viewing predicted class scores

## Technology stack

Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Pillow, and Streamlit.

## Architecture

Raw MRI Images
(train/test folders; four class subfolders)
          |
          v
Image Loading & Preprocessing
(224 x 224 resize; normalization; augmentation during training)
          |
          v
Model Training
  +--------------------+       +---------------------------+
  | Custom CNN         |       | MobileNetV2 Transfer      |
  | baseline           |       | learning + fine-tuning    |
  +--------------------+       +---------------------------+
              \                   /
               \                 /
                v               v
                 Model Evaluation
       (accuracy, precision, recall, F1,
          classification report, confusion matrix)
                          |
                          v
                  Model Comparison CSV
                          |
                          v
                    Streamlit App
             (upload MRI -> predicted class
                    and model scores)



## Project structure

```text
brain_tumor_mri_classification/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── raw/                  # Put dataset here; raw images are not included
├── models/                   # Trained models are generated here
├── outputs/                  # Metrics, plots, and comparison CSVs
└── src/
    ├── data_utils.py
    ├── train_cnn.py
    ├── train_transfer.py
    ├── evaluate.py
    └── compare_models.py
```

## Dataset arrangement

The scripts expect the following folder structure (class subfolders are discovered automatically):

```text
data/raw/
├── train/
│   └── train/
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
└── test/
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── no_tumor/
        └── pituitary/
```

The dataset is not bundled with this repository. Obtain it from a source you are permitted to use, follow that source's license/terms, and arrange the folders as above.

## Setup (Windows PowerShell)

Run commands from the project root.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the venv's Python directly:
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

## Train the Custom CNN

```powershell
python -m src.train_cnn --train-dir data/raw/train/train --test-dir data/raw/test/test --epochs 20
```

## Train MobileNetV2 transfer learning

```powershell
python -m src.train_transfer --train-dir data/raw/train/train --test-dir data/raw/test/test --epochs 8 --fine-tune-epochs 3
```

The MobileNetV2 model is saved in the native Keras `.keras` format to avoid legacy HDF5 serialization problems with preprocessing operations. The Custom CNN currently uses `.h5`.

## Compare models

```powershell
python -m src.compare_models
```

The comparison is saved to `outputs/model_comparison.csv`.

## Run the web app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`). Choose a model, upload a JPG/JPEG/PNG image, and review the predicted dataset class and scores.

## Current recorded results

These are the results recorded from the current run on 246 test images; retraining can produce different values.

| Metric | MobileNetV2 Transfer Learning | Custom CNN |
|---|---:|---:|
| Accuracy | 81.30% | 50.41% |
| Macro precision | 83.00% | 51.65% |
| Macro recall | 81.62% | 45.46% |
| Macro F1-score | 80.84% | 34.72% |
| Test images | 246 | 246 |

MobileNetV2 is the selected model based on these metrics. These scores are dataset-specific and do not establish clinical performance.

## Generated outputs

The `outputs/` directory can contain:

- `custom_cnn_metrics.json`
- `custom_cnn_classification_report.csv`
- `custom_cnn_confusion_matrix.png`
- `custom_cnn_history.png`
- `mobilenetv2_metrics.json`
- `mobilenetv2_classification_report.csv`
- `mobilenetv2_confusion_matrix.png`
- `mobilenetv2_history.png`
- `model_comparison.csv`

These files are generated by training/evaluation scripts and are not guaranteed to exist in a clean checkout until the scripts are run.

## Limitations and responsible use

- This project is for learning and demonstration only, not medical diagnosis.
- Evaluation used 246 test images; results may not generalize to other scanners, institutions, populations, or image protocols.
- A high model score is not a calibrated clinical probability.
- Check dataset licensing and attribution before redistributing any images.
- For real-world medical use, extensive external validation, clinical evaluation, governance, and regulatory review would be required.
