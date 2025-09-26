# BreastScan AI – Revolutionizing Breast Ultrasound Analysis with Deep Learning

## Project Overview
*BreastScan AI* is a deep learning–powered web application designed to assist healthcare professionals and researchers in analyzing breast ultrasound images. Using a Convolutional Neural Network (CNN), the system classifies scans into three categories: *Benign, **Malignant, or **Normal*.  

The tool provides *real-time predictions, confidence metrics, and downloadable reports through a **Flask-based web interface*. It’s intended for educational and research purposes, not for direct clinical diagnosis.  

---

## Table of Contents
1. [Abstract](#abstract)  
2. [Introduction](#introduction)  
3. [Objectives](#objectives)  
4. [Problem Statement](#problem-statement)  
5. [Literature Review](#literature-review)  
6. [System Design / Methodology](#system-design--methodology)  
7. [Implementation](#implementation)  
8. [Dataset Overview](#dataset-overview)  
9. [Results & Evaluation](#results--evaluation)  
10. [Discussion & Insights](#discussion--insights)  
11. [Conclusion & Future Scope](#conclusion--future-scope)  
12. [Deliverables](#deliverables)  
13. [References](#references)  

---

## Abstract
Breast cancer is among the most common and life-threatening diseases worldwide. Early detection significantly improves survival rates and treatment outcomes. This project, *BreastScan AI*, introduces a deep learning–powered web application designed to assist healthcare professionals and researchers in analyzing breast ultrasound images.  

The system leverages a CNN trained on the publicly available Breast Ultrasound Images Dataset (Kaggle) to classify scans into *Benign, Malignant, or Normal. The application provides **real-time predictions, confidence metrics, and downloadable reports* through an easy-to-use Flask interface.  

The model achieved *82% accuracy*, with balanced precision, recall, and F1-scores, demonstrating strong potential as an assistive tool for medical education and research.  

---

## Introduction
Breast cancer detection and diagnosis are critical areas in global healthcare. Traditional diagnostic methods, such as mammography and manual ultrasound interpretation, can be subjective and error-prone.  

*BreastScan AI* bridges this gap by combining AI with medical imaging to deliver an intelligent, user-friendly tool for automated breast ultrasound image classification. It supports healthcare professionals, researchers, and students in understanding breast abnormalities.  

---

## Objectives
- ✅ Develop a CNN-based AI model capable of classifying breast ultrasound images into *Benign, Malignant, or Normal* categories.  
- ✅ Provide a *user-friendly web application* for real-time image uploading and analysis.  
- ✅ Display results with *confidence metrics and recommendations* in an interpretable format.  
- ✅ Apply *data augmentation* techniques to balance the dataset and improve model performance.  
- ✅ Generate *downloadable reports* for each analysis, supporting medical education and research.  

---

## Problem Statement
Breast cancer is one of the most common cancers affecting women worldwide, and early detection improves treatment outcomes. Manual interpretation of ultrasound scans is prone to human error due to:  

- Low image quality  
- Radiologist fatigue  
- Variability in expertise  

Traditional methods lack automation and scalability, making *timely and accurate screening difficult. BreastScan AI provides a **reliable, AI-assisted classification system* as a second opinion for healthcare professionals.  

---

## Literature Review / Related Work
- CNNs outperform traditional image processing techniques in detecting tumors from medical scans.  
- Public datasets, like the *Breast Ultrasound Images Dataset (Kaggle)*, are commonly used.  
- Most prior works focus on *binary classification* (Benign vs Malignant). BreastScan AI uses a *three-class system*, improving diagnostic utility.  
- Existing solutions are often *research-only; BreastScan AI is deployed as a **web application* for practical usage.  

---

## System Design / Methodology
*Dataset:*  
- Source: Kaggle Breast Ultrasound Images Dataset  
- Categories: Benign, Malignant, Normal  
- Total Images: ~780  
- Data Augmentation: Applied using augmantation.py to balance classes  

*Data Preprocessing:*  
- Grayscale conversion  
- Resizing to 224×224 pixels  
- Normalization (0–1)  

*Model Development:*  
- Custom CNN with TensorFlow & Keras  
- Input: 224×224 grayscale images expanded to 3 channels  
- Output: 3 classes (Benign, Malignant, Normal)  

*Web Application:*  
- Backend: Flask (app.py)  
- Frontend: HTML, TailwindCSS, JavaScript  
- Features: Image upload, real-time prediction, confidence metrics, downloadable PDF report  

*System Architecture Diagram Placeholder*  

---

## Implementation
### Dataset Preparation
- Source: Kaggle  
- Classes: Benign, Malignant, Normal  
- Split: 70% Train, 15% Validation, 15% Test  
- Data augmentation applied via augmantation.py  

### Model Training
- Framework: TensorFlow & Keras  
- Architecture: Deep CNN with Batch Normalization & Dropout  
- Optimizer: Adam (lr=0.0001)  
- Loss: Categorical Cross-Entropy  
- Epochs: 20 (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)  

*CNN Layers:*  
- Conv2D → BatchNorm → MaxPooling (32, 64, 128, 256, 512 filters)  
- Flatten → Dense (512 + Dropout 0.5) → Dense (256 + Dropout 0.3) → Dense(3, Softmax)  

### Evaluation
- Accuracy: ~82%  
- Precision: 83%  
- Recall: 82%  
- F1 Score: 82%  

*Additional Evaluation:*  
- Confusion Matrix  
- ROC Curves & AUC  
- Classification Report  

### Deployment
- Flask backend (app.py)  
- Frontend: HTML + TailwindCSS + JS  
- Hosting: Local server (0.0.0.0:5000), future cloud deployment  

---

## Dataset Overview
The project uses the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) available on Kaggle.  

*Dataset Details:*  
- Total Images: 780  
- Classes:  
  - *Benign:* Non-cancerous tumors  
  - *Malignant:* Cancerous tumors  
  - *Normal:* Healthy tissue  
- Image Format: PNG  
- Size: ~500x500 pixels  
- Purpose: Used for training, validating, and testing the CNN model.  

*Download Dataset:* [Kaggle Dataset Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)  

---

## Results & Evaluation
| Metric | Value |
|--------|-------|
| Accuracy | 82% |
| Precision | 83% |
| Recall | 82% |
| F1 Score | 82% |

*Supporting Figures Placeholder:*  
- Training curves  
- Confusion Matrix  
- ROC Curves  

---

## Discussion & Insights
- Model shows *balanced performance* across classes  
- Limitations: Small dataset, grayscale images  
- Potential improvements: More data, advanced CNN architectures, Grad-CAM explainability  

---

## Conclusion & Future Scope
*Conclusion:*  
BreastScan AI demonstrates effective application of deep learning in breast ultrasound analysis. Real-time predictions and PDF report generation make it a *valuable educational and research tool*.  

*Future Scope:*  
- Improved accuracy with larger datasets  
- Advanced architectures (EfficientNet, ResNet, Transformers)  
- Explainability with Grad-CAM  
- Stage classification  
- Cloud deployment  

---

## Deliverables
- *Trained Model:* breast_cancer_cnn_best_one.keras  
- *Dataset:* Kaggle Breast Ultrasound Images Dataset  
- *Project Code:* breast.py, augmantation.py, app.py + HTML templates  
- *Documentation:* PDF report  
- *Presentation Slides*  
- *Demo:* Flask web app  

---

## References
- Kaggle Dataset – Arya Shah: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)  
- TensorFlow Documentation: [https://www.tensorflow.org](https://www.tensorflow.org)  
- Keras Documentation: [https://keras.io](https://keras.io)  
- Yala, A. et al. (2019). Deep Learning for Breast Cancer Detection. Nature Medicine
