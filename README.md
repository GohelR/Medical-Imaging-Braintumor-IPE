# 🧠 Brain MRI Tumor Detection — Medical Imaging Data Science Project

A comprehensive deep learning project for detecting brain tumors from MRI scans using **Custom CNN** and **EfficientNetB0 Transfer Learning**, developed as part of the **Introduction to Prompt Engineering (IPE)** subject at **Marwadi University**.

---

## 👨‍💻 Author Information
**Name:** Ravi Gohel  
**Branch:** B.Tech (CSE – AI & ML)  
**University:** Marwadi University  
**Academic Year:** 2025  

---

## 🎯 Project Overview

This project focuses on building and comparing **two deep learning models** for binary brain tumor detection:
1. 🧩 **Custom CNN** — a handcrafted lightweight Convolutional Neural Network.  
2. ⚙️ **EfficientNetB0 Transfer Learning** — a pretrained model fine-tuned for medical imaging.

The objective is to evaluate both models on accuracy, recall, and clinical usefulness, demonstrating how AI can support radiologists in early tumor detection.

---

## 🔍 Key Features
- ✅ Data preprocessing & augmentation pipeline  
- ✅ Class imbalance handling using weighted loss  
- ✅ CNN vs. Transfer Learning comparison  
- ✅ Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- ✅ Confusion matrix & Grad-CAM visualization  
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
- **Image Size:** 224×224 pixels  

---

## ⚙️ Model Architectures

### 🧠 Custom CNN
A lightweight 3-block CNN designed for binary classification.

Input (224, 224, 3)
│
├── Conv2D(32) → ReLU → MaxPooling2D
├── Conv2D(64) → ReLU → MaxPooling2D
├── Conv2D(128) → ReLU → MaxPooling2D
│
├── Flatten
├── Dense(128) → Dropout(0.5)
└── Dense(1, activation='sigmoid')


**Parameters:** ~44M  
**Optimizer:** Adam  
**Loss:** Binary Cross-Entropy  
**Regularization:** Dropout (0.5)  

---

### ⚡ EfficientNetB0 (Transfer Learning)
Pretrained on ImageNet and fine-tuned on the MRI dataset.

Base Model: EfficientNetB0 (frozen base layers)
│
├── GlobalAveragePooling2D
├── Dense(256, ReLU) → Dropout(0.4)
├── Dense(128, ReLU) → Dropout(0.3)
└── Dense(1, activation='sigmoid')


**Parameters:** ~25M  
**Optimizer:** Adam (LR=1e-5)  
**Loss:** Binary Cross-Entropy  
**Technique:** Two-phase training (feature extraction + fine-tuning)  

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
Weighted loss applied for class imbalance
Augmentation improves generalization
---
**📈 Model Evaluation**
CNN Results
| Metric    | Train | Validation |
| --------- | ----- | ---------- |
| Accuracy  | 81%   | 72%        |
| Precision | 73%   | 74%        |
| Recall    | 87%   | **82%**    |
| F1-score  | 79%   | 77%        |

✅ Strong recall for “Tumor” class — ideal for medical screening.
⚠️ Slight drop in No-Tumor accuracy (class imbalance impact).
---
**EfficientNetB0 Results.**
| Metric   | Train                           | Validation    |
| -------- | ------------------------------- | ------------- |
| Accuracy | 52%                             | 62%           |
| Recall   | 100% (Tumor)                    | 0% (No Tumor) |
| Issue    | Predicted all images as “Tumor” |               |

Observation: Transfer model overfitted due to small dataset and failed to generalize.
---
**📊 Visual Results**

🧩 Confusion Matrix: Showed high true positives for Tumor class.
🔵 ROC Curve: AUC ≈ 0.86 for CNN.
🔥 Grad-CAM: Highlights regions of MRI that influenced prediction.
Visual Insights:
The CNN focused correctly on tumor regions, proving interpretability.
---
**🩻 Clinical Discussion**

🩺 Use Case: AI-assisted screening for radiologists.
🕐 Impact: Speeds up triage for tumor detection.
✅ Advantage: High recall ensures fewer missed tumor cases.
⚠️ Limitation: Must be verified by human experts before deployment.
---
**🔮 Future Improvements**

Collect larger MRI datasets from multiple hospitals.
Implement Cross-Validation for robust results.
Extend to multi-class classification (Glioma, Meningioma, Pituitary).
Integrate Explainable AI (Grad-CAM, LIME) visual tools.
Deploy a Streamlit-based Diagnostic Web App for clinicians.
---
**🧰 Installation & Usage**
🪄 Environment Setup
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python jupyter

**🧪 Run the Notebook**
jupyter notebook notebook/ipe-project-work.ipynb
**
🧠 Run in Kaggle or Colab**
Simply upload the notebook to Google Colab or Kaggle and execute all cells — no local setup required.
---
**📜 Results Summary**
| Model               | Accuracy | Recall (Tumor)    | F1-score | Status       |
| ------------------- | -------- | ----------------- | -------- | ------------ |
| Logistic Regression | 76.4%    | -                 | 75%      | Baseline     |
| Custom CNN          | 72%      | **87%**           | 77%      | ✅ Best Model |
| EfficientNetB0      | 62%      | 100% (only Tumor) | -        | ❌ Overfit    |
---
---
**🧾 License**
This project is licensed under the MIT License.
---
**📚 References**

Navoneel Chakrabarty — Brain MRI Images for Brain Tumor Detection, Kaggle Dataset
Tan & Le — EfficientNet: Rethinking Model Scaling for CNNs, ICML 2019
Ronneberger et al. — U-Net for Biomedical Image Segmentation, MICCAI 2015
Chollet, F. — Deep Learning with Python, Manning, 2018
TensorFlow & Keras Official Documentation
**
---⚠️ Disclaimer**

This software is intended for academic and research purposes only.
It is not a certified medical diagnostic system and should never replace professional medical advice or radiologist interpretation.
------
**📞 Contact**

📧 Ravi Gohel - ravi.n.gohel811@gmail.com
🏫 Marwadi University - Department of Computer Science & Engineering (AI & ML)
🧠 Project under Introduction to Prompt Engineering (IPE)
---
**Made with ❤️ to advance AI in healthcare**
