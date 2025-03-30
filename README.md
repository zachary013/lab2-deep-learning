# Lab 2: Computer Vision with PyTorch 🖼️

![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN%20%26%20Transformers-blue?style=for-the-badge&logo=python)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)  
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensource)  
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0.1-orange?style=for-the-badge&logo=pytorch)

## 🔥 Project Overview
Welcome to **Lab 2: Computer Vision with PyTorch**, my submission for the Deep Learning course at Abdelmalek Essaadi University! This project tackles MNIST digit classification using a diverse set of models:
- **CNN**: A custom convolutional neural network for efficient classification.
- **ResNet50**: Adapted from Faster R-CNN’s backbone, fine-tuned for MNIST.
- **VGG16 & AlexNet**: Pre-trained models fine-tuned for digit recognition.
- **Vision Transformer (ViT)**: A transformer built from scratch to explore modern architectures.

Built with **PyTorch** on Google Colab’s T4 GPU, this repo includes model implementation, training, fine-tuning, and performance comparisons—all in a single Jupyter Notebook.

---

## 🛠️ Features
- 🖼️ **MNIST Classification**: Classifies digits using 5 models (CNN, ResNet50, VGG16, AlexNet, ViT).
- 📊 **Comprehensive Metrics**: Evaluates accuracy, F1 score, loss, and training time.
- ⚡ **GPU Acceleration**: Utilizes Colab’s T4 GPU for faster training.
- 🔧 **Fine-Tuning**: Adjusts pre-trained VGG16 and AlexNet for MNIST (1-channel, 10 classes).
- 🧠 **From Scratch**: Implements ViT with patch embeddings and transformer layers.

---

## 📋 Table of Contents
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Prepared by](#prepared-by)
- [Contributing](#-contributing)
- [License](#-license)
- [Shoutouts](#-shoutouts)

---

## 📦 Requirements
To run this project, you'll need:
- **Python 3.8+**
- **Jupyter Notebook** (or Colab)
- **Dependencies**:
  ```plaintext
  torch==2.0.1
  torchvision==0.15.2
  numpy==1.24.3
  scikit-learn==1.3.0
  ```
Optional: GPU with CUDA for faster training.

## ⚙️ Installation
Clone the Repo:
```bash
git clone https://github.com/zachary013/lab2-computer-vision.git
cd lab2-computer-vision
```

Set Up a Virtual Env:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install the Dependencies:
```bash
pip install -r requirements.txt
```
No requirements.txt yet? Copy the list above and run `pip install <package>` for each.

Launch Jupyter:
```bash
jupyter notebook
```

## 🚀 Usage
Open the Notebook:
Launch `lab2_computer_vision.ipynb` in Jupyter or upload it to Google Colab.

Run It:
Click "Run All" (or Shift + Enter cell-by-cell) to:
- Download the MNIST dataset via torchvision.
- Train and evaluate all 5 models.
- Display metrics (accuracy, F1 score, loss, training time).

Outputs:
View inline results for each model’s performance.

Pro Tip: Use Colab's free T4 GPU for faster training—set it up in Runtime > Change runtime type!

## 📂 Project Structure
```text
lab2-computer-vision/
├── lab2_computer_vision.ipynb  # Main notebook with code and results
├── lab2_report.pdf             # LaTeX report with detailed analysis
├── requirements.txt            # Dependency list (create if missing)
└── README.md                   # This guide
```
MNIST dataset is fetched dynamically via torchvision—no local storage needed!

## 🎯 Results
### Model Performance
All models were trained on MNIST (60,000 train, 10,000 test images) using Colab's T4 GPU. Results to be updated after running the notebook:

**CNN**:
- Accuracy: [TBD]%
- F1 Score: [TBD]
- Loss: [TBD]
- Time: [TBD]s

**ResNet50**:
- Accuracy: [TBD]%
- F1 Score: [TBD]
- Loss: [TBD]
- Time: [TBD]s

**VGG16**:
- Accuracy: [TBD]%
- F1 Score: [TBD]
- Loss: [TBD]
- Time: [TBD]s

**AlexNet**:
- Accuracy: [TBD]%
- F1 Score: [TBD]
- Loss: [TBD]
- Time: [TBD]s

**ViT**:
- Accuracy: [TBD]%
- F1 Score: [TBD]
- Loss: [TBD]
- Time: [TBD]s

### Key Takeaways
- Pre-trained models (VGG16, AlexNet, ResNet50) are expected to lead due to transfer learning.
- CNN should be the fastest, ideal for simple tasks like MNIST.
- ViT, built from scratch, may take longer but showcases transformer potential.
- ResNet50 adapts Faster R-CNN’s backbone effectively for classification.

## Prepared by
| Avatar | Name | GitHub |
|--------|------|--------|
| <img src="https://github.com/zachary013.png" width="50" height="50" style="border-radius: 50%"/> | Zakariae Azarkan | @zachary013 |

## 🤝 Contributing
This is a university assignment, so I’m not accepting pull requests. Feel free to fork and experiment, though! Have suggestions? Share them in GitHub Issues.

## 📜 License
Licensed under the MIT License—use it, share it, just don’t blame me if your GPU gets too hot! 🔥

## 🙌 Shoutouts
- Prof. Lotfi Elaachak for guiding this lab.
- PyTorch team for an amazing framework.
- Google Colab for free GPU access—a true lifesaver!
- MNIST Dataset for being the perfect playground for computer vision tasks.
