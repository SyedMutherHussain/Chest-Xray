# Chest-Xray
Generative Adversarial network trained pneumonia chest x-ray diagnosis application.
# GAN-based Image Augmentation for Medical Imaging 🏥

## Pediatric Chest X-ray Pneumonia Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)


> **⚠️ DISCLAIMER: This project is for educational and research purposes only. Not intended for actual medical diagnosis.**

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an automated pneumonia detection system for pediatric chest X-rays using a novel combination of **Convolutional Neural Networks (CNN)** and **Generative Adversarial Networks (GAN)**. The GAN component addresses the common challenge of limited medical imaging data through synthetic image generation, while the CNN performs accurate binary classification.

### 🎨 Key Innovation
- **GAN-based Data Augmentation**: Generates realistic synthetic chest X-rays to expand training dataset
- **Robust CNN Architecture**: Optimized for medical image classification with dropout and batch normalization
- **User-Friendly Interface**: Web-based application for easy image upload and analysis

## ✨ Features

- 🖼️ **Multi-format Support**: Accepts JPEG, PNG, TIFF, and DICOM image formats
- ⚡ **Real-time Analysis**: Instant classification with confidence scores
- 📊 **Performance Metrics**: Detailed accuracy, precision, and recall reporting
- 🎓 **Educational Focus**: Designed for learning and research applications
- 🌐 **Web Interface**: No installation required for end users
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.18% |
| **Precision** | 0.87 |
| **Recall** | 0.87 |
| **Input Size** | 148×148 pixels |

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Clone Repository
```bash
git clone https://github.com/yourusername/gan-medical-imaging.git
cd gan-medical-imaging
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
opencv-python>=4.5.0
streamlit>=1.10.0
pillow>=9.0.0
scikit-learn>=1.0.0
```

## 🚀 Usage

### 1. Train the Model
```bash
# Train GAN for data augmentation
python train_gan.py --epochs 100 --batch_size 32

# Train CNN classifier
python train_classifier.py --epochs 50 --augment True
```

### 2. Run Web Application
```bash
streamlit run app.py
```

### 3. Use the Interface
1. Open your browser and navigate to `http://localhost:8501`
2. Upload a chest X-ray image (any format)
3. Click **"Analyze X-ray"**
4. View results with confidence scores

### 4. Command Line Prediction
```bash
python predict.py --image path/to/xray.jpg
```

## 📊 Dataset

This project uses publicly available pediatric chest X-ray datasets:
- **Training Images**: 5,232 chest X-rays
- **Test Images**: 624 chest X-rays  
- **Classes**: Normal (0) and Pneumonia (1)
- **Age Group**: Pediatric patients (1-5 years)

### Data Preprocessing
- Resize to 148×148 pixels
- Normalization (0-1 range)
- GAN augmentation (2x dataset expansion)

## 🏗️ Model Architecture

### GAN Architecture
```
Generator: Dense → Reshape → Conv2DTranspose × 4 → Tanh
Discriminator: Conv2D × 4 → Flatten → Dense → Sigmoid
```

### CNN Architecture
```
Conv2D(32) → BatchNorm → ReLU → MaxPool
Conv2D(64) → BatchNorm → ReLU → MaxPool  
Conv2D(128) → BatchNorm → ReLU → MaxPool
Flatten → Dense(128) → Dropout(0.5) → Dense(1) → Sigmoid
```

## 📁 Project Structure

```
gan-medical-imaging/
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── gan_generator.h5
│   ├── gan_discriminator.h5
│   └── cnn_classifier.h5
├── src/
│   ├── gan_model.py
│   ├── cnn_model.py
│   ├── preprocessing.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── app.py
├── train_gan.py
├── train_classifier.py
├── predict.py
├── requirements.txt
└── README.md
```

## 📊 Results

### Training History
- **GAN Training**: 100 epochs, stable loss convergence
- **CNN Training**: 50 epochs, early stopping implemented
- **Validation Accuracy**: 85.3%
- **Test Accuracy**: 87.18%

### Sample Predictions
| Image | Prediction | Confidence |
|-------|------------|------------|
| Normal X-ray | Normal | 92.3% |
| Pneumonia X-ray | Pneumonia | 89.7% |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- 🧠 Model architecture improvements
- 📊 Additional evaluation metrics
- 🌐 UI/UX enhancements
- 📚 Documentation improvements
- 🧪 Unit tests

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Medical imaging dataset providers
- TensorFlow and Keras communities
- Healthcare professionals for domain expertise
- Open source contributors

## 📞 Contact

- **Author**: Syed Muther Hussain
- **Email**: syedmutherhussain786@gmail.com
- **LinkedIn**: linkedin.com/in/syed-muther-hussain-a-587b94256
- **Project Link**: https://github.com/SyedMutherHussain/Chest-Xray/edit/main/README.md

## ⚡ Quick Start

```bash
# One-command setup
git clone https://github.com/yourusername/gan-medical-imaging.git
cd gan-medical-imaging
pip install -r requirements.txt
streamlit run app.py
```

---

**⭐ Star this repository if you found it helpful!**
