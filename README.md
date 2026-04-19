# Emotion Recognition Web — FER-2013 Mobile FESNet

A real-time facial emotion recognition system built with a deep learning pipeline (IR-SE50 backbone + MLP classifier) and deployed as a browser-based web application using TensorFlow.js and face-api.js.

---

## Overview

This project implements an end-to-end facial emotion recognition pipeline. A pretrained **IR-SE50** model (InsightFace) extracts 512-dimensional face embeddings, which are then classified by a lightweight **MLP classifier** trained on the **FER-2013** dataset. The trained model is exported through a PyTorch → ONNX → TensorFlow.js conversion chain and served as a fully client-side web application — no backend server required.

### Recognized Emotions

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## System Architecture

```
[Input Image / Webcam]
        ↓
[Face Detection — MTCNN / face-api.js]
        ↓
[Feature Extraction — IR-SE50 Backbone (512-dim embedding)]
        ↓
[Emotion Classification — Keras MLP]
        ↓
[Model Conversion Pipeline]
  PyTorch (.pth) → ONNX → TensorFlow SavedModel → TensorFlow.js
        ↓
[Web Application — face-api.js real-time inference in browser]
        ↓
[Output: Bounding Box + Emotion Label]
```

---

## Dataset

**FER-2013** — Facial Expression Recognition 2013

| Property | Detail |
|---|---|
| Source | [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013/data) |
| Image size | 48 × 48 pixels (grayscale) |
| Training samples | 28,709 images |
| Test samples | 3,589 images |
| Classes | 7 emotion categories |

Images are centered and preprocessed, with augmentation applied to minority classes (disgust, fear, surprise) to address class imbalance.

---

## Model Pipeline

### Stage 1 — Feature Extraction (IR-SE50)

- Architecture: ResNet-50-based backbone with **Squeeze-and-Excitation (SE)** modules
- Pretrained on face recognition (InsightFace weights)
- Outputs: L2-normalized 512-dimensional face embeddings
- Key components:
  - `SEModule` — adaptive channel attention via global average pooling
  - `BottleneckIRSE` — residual blocks with PReLU activation and skip connections
  - `Backbone` — 50-layer network producing fixed-size embeddings

### Stage 2 — Emotion Classifier (Keras MLP)

- Input: 512-dim embedding from IR-SE50
- Architecture: Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.5) → Dense(64) → Dropout(0.5) → Dense(7, softmax)
- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Training: 80/20 train-validation split, class weights for imbalance, early stopping (patience=10)
- Saved as: `emotion_model.h5`

### Stage 3 — Model Conversion

```
PyTorch .pth
    ↓  torch.onnx.export (opset 11, dynamic batch)
ONNX ir_se50.onnx
    ↓  onnx-tf prepare + export_graph
TensorFlow SavedModel
    ↓  tensorflowjs_converter
TensorFlow.js (model.json + .bin shards)
```

Conversion is performed inside a **Docker container** (Python 3.10-slim) with pinned library versions to ensure reproducibility:

```
numpy==1.23.5 | onnx==1.12.0 | onnxruntime==1.12.1
onnx-tf==1.9.0 | tensorflow==2.10.1 | tensorflowjs==3.18.0
torch==1.13.1+cpu | torchvision==0.14.1+cpu
```

---

## Web Application

The front-end is a single HTML file using **face-api.js** for browser-native face detection and emotion recognition. No backend or Python environment is needed at inference time.

### Features

- **Webcam Detection** — real-time emotion recognition from live camera feed
- **Photo Upload** — static image analysis with bounding box overlay
- **Visual Feedback** — red bounding box drawn over detected face with dominant emotion label

### How It Works

1. `loadModels()` — loads SSD MobileNetV1 (face detection), FaceLandmark68Net, and FaceExpressionNet from `./models/`
2. `startCamera()` — requests webcam access via `navigator.mediaDevices.getUserMedia`
3. `detectLoop()` — runs `requestAnimationFrame` loop; detects face, extracts landmarks + expressions, renders bounding box, displays dominant emotion
4. Upload handler — processes static images through the same detection pipeline on demand

---

## Results

### Training Performance (15 epochs)

| Metric | Value |
|---|---|
| Training accuracy (final) | ~0.65 |
| Validation accuracy | ~0.51 |
| Test accuracy | ~0.50 |
| Embedding dimension | 512 |
| Training samples used | 3,436 |
| Test samples used | 700 |

Loss decreased consistently across epochs without significant overfitting, indicating stable convergence. The gap between training and validation accuracy reflects the inherent difficulty of the FER-2013 dataset and the limited data used for speed optimization. Further improvement is possible with more training data, fine-tuning the backbone, or using a deeper classifier.

### Inference Results

Live webcam testing on 40 random images demonstrated successful end-to-end inference. MTCNN face detection occasionally failed on low-resolution or poorly-lit frames; in such cases, the system falls back to using the full image as input.

---

## Project Structure

```
MDI_EmotionRecognitionWEB/
├── index.html                  # Main web application (single file)
├── models/                     # face-api.js model weights
│   ├── ssd_mobilenetv1/
│   ├── face_landmark_68/
│   └── face_expression/
├── irse50_converter/           # Docker-based conversion environment
│   ├── Dockerfile
│   ├── export_ir_se50_to_onnx.py
│   └── onnx_to_savedmodel.py
└── output/
    └── ir_se50_tfjs/
        ├── model.json          # TensorFlow.js model architecture
        └── group1-shard*.bin   # Weight shards (~4 KB each, 42 files)
```

---

## Getting Started

### Prerequisites

- Modern web browser (Chrome, Firefox, Edge — webcam API required)
- A local HTTP server (required for loading model files — `file://` protocol will be blocked by CORS)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/zXmill/MDI_EmotionRecognationWEB.git
cd MDI_EmotionRecognationWEB

# Serve with any static server, for example:
python -m http.server 8080
# or
npx serve .
```

Then open `http://localhost:8080` in your browser.

### Run Model Conversion (Docker)

```bash
# Build the converter image
cd irse50_converter
docker build -t irse50-converter .

# Export IR-SE50 to ONNX (place your .pth weights in the directory first)
docker run --rm -v $(pwd):/work irse50-converter \
  python export_ir_se50_to_onnx.py model_ir_se50.pth ir_se50.onnx

# Convert ONNX to TensorFlow SavedModel
docker run --rm -v $(pwd):/work irse50-converter \
  python onnx_to_savedmodel.py ir_se50.onnx ir_se50_saved_model

# Convert SavedModel to TensorFlow.js
tensorflowjs_converter \
  --input_format=tf_saved_model \
  ir_se50_saved_model \
  output/ir_se50_tfjs
```

---

## Training (Google Colab / Kaggle)

Install dependencies:

```bash
pip install tensorflow keras facenet-pytorch opencv-python scikit-learn tensorflowjs torch numpy kagglehub matplotlib
```

The training pipeline runs in 8 steps:

1. Install packages and imports
2. Download and extract FER-2013 dataset
3. Preprocessing with augmentation for minority classes
4. Load dataset in batches (max 500 train / 100 test per class for speed)
5. Extract IR-SE50 embeddings (512-dim)
6. Train Keras MLP classifier with class weights and early stopping
7. Evaluate and save `emotion_model.h5`
8. Convert to TensorFlow.js format

---

## Technology Stack

| Component | Technology |
|---|---|
| Feature extractor | IR-SE50 (InsightFace, PyTorch) |
| Classifier | Keras MLP (TensorFlow) |
| Face detection (training) | MTCNN (facenet-pytorch) |
| Face detection (web) | face-api.js (SSD MobileNetV1) |
| Model conversion | ONNX, onnx-tf, TensorFlow.js |
| Containerization | Docker (python:3.10-slim) |
| Web frontend | Vanilla HTML/CSS/JavaScript |
| Dataset | FER-2013 (Kaggle) |

---

## Known Limitations

- MTCNN may fail to detect faces on very small (48×48 FER-2013) images; the system falls back to using the full image
- Test accuracy (~50%) reflects the difficulty of generalizing on FER-2013 with limited training samples per class
- The `.h5` model format is considered legacy by Keras; migrating to `.keras` format is recommended for future versions
- HDF5 save warnings are non-critical and do not affect model conversion or inference

## Future Improvements

- Fine-tune the IR-SE50 backbone end-to-end on FER-2013 for improved accuracy
- Replace MLP with a deeper classifier or attention-based head
- Add multi-face detection and per-face emotion tracking
- Improve MTCNN reliability on low-resolution inputs
- Migrate model saving to `.keras` format

---

## License

This project is submitted as academic coursework for **Manajemen Data dan Informasi** at Universitas Negeri Surabaya. Educational use only.

---

## Acknowledgments

- **InsightFace / IR-SE50** pretrained weights: [lithiumice/insightface](https://huggingface.co/lithiumice/insightface)
- **FER-2013 Dataset**: [msambare on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)
- **face-api.js**: [vladmandic/face-api](https://github.com/vladmandic/face-api)
