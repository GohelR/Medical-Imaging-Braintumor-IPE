# 🧠 Brain MRI Tumor Detection — Medical Imaging Data Science Project

A comprehensive deep learning project for detecting brain tumors from MRI scans using **Custom CNN** and **EfficientNetB0 Transfer Learning**, developed as part of the **Introduction to Prompt Engineering (IPE)** subject at **Marwadi University**.

---

## 👨‍💻 Author
**Name:** Ravi Gohel  
**Branch:** B.Tech (CSE – AI & ML)  
**University:** Marwadi University  
**Academic Year:** 2025

---

## 🎯 Project Overview
This project builds and compares **two deep learning models** for binary brain tumor detection:

1. 🧩 **Custom CNN** — a handcrafted lightweight Convolutional Neural Network.  
2. ⚙️ **EfficientNetB0 (Transfer Learning)** — a pretrained model fine-tuned for medical imaging.

The goal is to evaluate both models for accuracy, recall, and clinical usefulness, showing how AI can assist radiologists in early tumor detection.

---

## 🔍 Key Features
- ✅ Data preprocessing & augmentation pipeline  
- ✅ Class imbalance handling using weighted loss  
- ✅ CNN vs. Transfer Learning comparison  
- ✅ Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- ✅ Confusion matrix & Grad-CAM visualizations  
- ✅ Report + Presentation included  
- ✅ Clinical implications discussion  

---

## 📊 Dataset Details
- **Source:** [Kaggle — Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
- **Classes:** Tumor, No Tumor  
- **Image Type:** MRI (Grayscale)  
- **Total Samples:** 253 images  
  - Tumor: 155  
  - No Tumor: 98  
- **Split:** 80% Train / 20% Validation  
- **Image Size:** 224 × 224 pixels  

---

## ⚙️ Model Architectures

### 🧠 Custom CNN
Input (224, 224, 3)
├── Conv2D(32) → ReLU → MaxPool
├── Conv2D(64) → ReLU → MaxPool
├── Conv2D(128) → ReLU → MaxPool
├── Flatten
├── Dense(128) → Dropout(0.5)
└── Dense(1, activation='sigmoid')

yaml
Copy code
- **Parameters:** ~44M  
- **Optimizer:** Adam  
- **Loss:** Binary Cross-Entropy  

---

### ⚡ EfficientNetB0 (Transfer Learning)
EfficientNetB0 base (frozen initially)
├── GlobalAveragePooling2D
├── Dense(256, ReLU) → Dropout(0.4)
├── Dense(128, ReLU) → Dropout(0.3)
└── Dense(1, activation='sigmoid')

yaml
Copy code
- **Parameters:** ~25M  
- **Optimizer:** Adam (LR=1e-5)  
- **Loss:** Binary Cross-Entropy  
- **Technique:** Two-phase training (feature extraction + fine-tuning)

---

## 🧩 Data Preprocessing & Augmentation
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
Images resized to 224×224

Normalized to [0,1]

Weighted loss for class imbalance

Augmentation improves generalization

📈 Model Evaluation
🧠 Custom CNN
Metric	Train	Validation
Accuracy	81%	72%
Precision	73%	74%
Recall	87%	82%
F1-score	79%	77%

✅ Strong recall for “Tumor” class — ideal for screening.
⚠️ Slight drop in No-Tumor accuracy due to class imbalance.

⚡ EfficientNetB0
Metric	Train	Validation
Accuracy	52%	62%
Recall	100% (Tumor only)	0% (No Tumor)

⚠️ Observation: The model predicted all samples as “Tumor,” indicating overfitting from limited data.

📊 Visual Results
🧩 Confusion Matrix: High true positives for Tumor class

🔵 ROC Curve: AUC ≈ 0.86 for CNN

🔥 Grad-CAM: Highlights tumor regions that influenced predictions

Visual insights confirm CNN’s interpretability and reliability.

🩻 Clinical Discussion
🩺 Use Case: AI-assisted screening for radiologists

🕐 Impact: Speeds up triage for tumor detection

✅ Advantage: High recall ensures fewer missed cases

⚠️ Limitation: Requires validation on larger datasets before clinical use

🔮 Future Improvements
Collect larger MRI datasets from multiple hospitals

Implement Cross-Validation for robust performance

Extend to multi-class classification (Glioma, Meningioma, Pituitary)

Add Explainable AI (Grad-CAM, LIME)

Deploy using Streamlit for radiologist workflow

🧰 Installation & Usage
🪄 Environment Setup
bash
Copy code
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python jupyter
🧪 Run the Notebook
bash
Copy code
jupyter notebook notebook/ipe-project-work.ipynb
💡 Or run directly on Google Colab / Kaggle — no setup required.

📜 Results Summary
Model	Accuracy	Recall (Tumor)	F1-score	Status
Logistic Regression	76.4%	-	75%	Baseline
Custom CNN	72%	87%	77%	✅ Best Model
EfficientNetB0	62%	100% (only Tumor)	-	❌ Overfit

🧾 License
This project is licensed under the MIT License — see the LICENSE file for details.

📚 References
Navoneel Chakrabarty — Brain MRI Images for Brain Tumor Detection, Kaggle Dataset

Tan & Le — EfficientNet: Rethinking Model Scaling for CNNs, ICML 2019

Ronneberger et al. — U-Net for Biomedical Image Segmentation, MICCAI 2015

Chollet, F. — Deep Learning with Python, Manning, 2018

TensorFlow / Keras Official Documentation

⚠️ Disclaimer
This software is for academic and research purposes only.
It is not a certified medical diagnostic system and should not replace professional radiologist evaluation.

📞 Contact
📧 Email: ravi.n.gohel811@gmail.com
🏫 Institute: Marwadi University
🎓 Department: Computer Science & Engineering (AI & ML)
🧠 Subject: Introduction to Prompt Engineering (IPE)

Made with ❤️ to advance AI in Healthcare
