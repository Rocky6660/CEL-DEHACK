# CEL-DEHACK

Here is a professional `README.md` file tailored for your GitHub repository. It incorporates the "DeHack" narrative and technical details from your code.

---

# Automated Retail Checkout System (Vista Challenge)

**Team:** 404 Brain not found

**Event:** DeHack 2026

## 📖 Overview

This project is a computer vision solution designed to "see like a cashier." The goal is to automate the retail checkout process by accurately detecting, identifying, and counting products (such as puffed food, dairy, and stationery) from images, even in cluttered real-world scenarios.

Instead of relying solely on the provided raw dataset—which presented significant data formatting and quality challenges—we engineered a **Synthetic Data Generation Pipeline** to create a robust, noise-free training set for a YOLOv8 model.

## 🚀 Key Features

* **Synthetic Data Pipeline:** Custom image generation using OpenCV to crop product instances and paste them onto random backgrounds, creating 1,500 distinct training samples.
* **Object Detection:** Utilizes **YOLOv8s (Small)** architecture for an optimal balance of inference speed and detection accuracy.
* **Large-Scale Inference:** Capable of processing and generating predictions for over 18,000 test images efficiently.
* **Scalable Categories:** Configured to handle 200 distinct product categories.

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Core Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Computer Vision:** OpenCV (`cv2`), Albumentations
* **Data Handling:** NumPy, Pandas
* **Environment:** Jupyter Notebook / Kaggle Kernel

## ⚙️ Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install ultralytics albumentations opencv-python numpy pandas matplotlib tqdm

```

## 🧠 Methodology & Workflow

### 1. The Bottleneck (Raw Data)

The initial attempt involved training directly on the raw real-world dataset. However, converting the complex raw annotations into the YOLO format proved to be error-prone and inefficient, leading to poor model convergence.

### 2. The Solution (Synthetic Generation)

To overcome data limitations, we built a synthetic data generator:

* **Extraction:** Valid object instances were cropped from source training images based on their bounding box coordinates.
* **Synthesis:** These crops were resized (random scaling 0.3x - 0.8x) and pasted onto random background templates to create new, perfectly annotated images.
* **Output:** Generated 1,500 synthetic images (80% train / 20% val) with corresponding YOLO-formatted labels.

### 3. Model Training

* **Model:** YOLOv8s (`yolov8s.pt`)
* **Epochs:** 100
* **Image Size:** 640x640
* **Optimizer:** AdamW (`lr=0.0005`, `momentum=0.9`)

### 4. Inference

The trained model weights (`best.pt`) are loaded to run inference on the test dataset. The results are formatted into a submission CSV containing `ImageID` and `PredictionString` (class, confidence, coordinates).

## 📊 Results

* **Training mAP50:** ~0.653
* **Training mAP50-95:** ~0.575
* **Test Images Processed:** 18,000

## 📂 Project Structure

```
├── cel-dehack.ipynb       # Main notebook containing generation, training, and inference code
├── submission.csv         # Final output file with predictions
├── runs
├── synthetic_dataset/     # Generated training data (created at runtime)
│   ├── images/
│   └── labels/
├── yolo26n.pt
├── yolo8s.pt
└── submission.csv


```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

*Built with 💻 and ☕ by Team 404 Brain not found.*
