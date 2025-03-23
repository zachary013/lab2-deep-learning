# Lab 2: Computer Vision with PyTorch 🖼️

![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN%20%26%20Transformers-blue?style=for-the-badge&logo=python)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)  
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensource)  
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0.1-orange?style=for-the-badge&logo=pytorch)

## 🔥 Project Overview
Welcome to **Lab 2: Computer Vision with PyTorch**, my submission for the Deep Learning course at Abdelmalek Essaadi University! This project dives into MNIST digit classification using a lineup of powerful models:
- **CNN**: A custom convolutional neural network for fast and effective classification.
- **Faster R-CNN**: Adapted for classification, showing off object detection capabilities.
- **VGG16 & AlexNet**: Pretrained giants fine-tuned for MNIST.
- **Vision Transformer (ViT)**: A from-scratch transformer to flex modern architectures.

Built with **PyTorch** on Google Colab’s T4 GPU, this repo packs a punch with model building, fine-tuning, and metric comparisons—all in a single Jupyter Notebook.

---

## 🛠️ Features
- 🖼️ **MNIST Classification**: Tackles digit recognition with 5 models (CNN, Faster R-CNN, VGG16, AlexNet, ViT).
- 📊 **Metrics Galore**: Accuracy, F1 score, loss, and training time for all models.
- ⚡ **GPU Power**: Leverages Colab’s T4 GPU for speedy training.
- 🔧 **Fine-Tuning**: Pretrained VGG16 and AlexNet adjusted for MNIST (1-channel, 10 classes).
- 🧠 **From Scratch**: ViT built from the ground up—patches, transformers, and all!

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
  pandas==2.0.3
  numpy==1.24.3
  scikit-learn==1.3.0
  kagglehub
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

Install the Goods:
```bash
pip install -r requirements.txt
```
No requirements.txt yet? Copy the list above and run pip install <package> for each.

Launch Jupyter:
```bash
jupyter notebook
```

## 🚀 Usage
Open the Notebook:
Fire up lab2_computer_vision.ipynb in Jupyter or upload it to Google Colab.

Run It:
Hit Run All (or Shift + Enter cell-by-cell) to:
- Download the MNIST dataset via torchvision.
- Train and evaluate all 5 models.
- Print metrics (accuracy, F1, loss, time).

Outputs:
Check inline results for each model's performance.

Pro Tip: Use Colab's free T4 GPU for faster training—set it up in Runtime > Change runtime type!

## 📂 Project Structure
```text
lab2-computer-vision/
├── lab2_computer_vision.ipynb  # The main notebook: code and results
├── lab2_report.pdf             # LaTeX report with detailed analysis
├── requirements.txt            # Dependency list (create it if missing)
└── README.md                   # This guide
```
MNIST dataset is fetched dynamically via torchvision—no local storage needed!

## 🎯 Results
### Model Performance
All models were trained on MNIST (60,000 train, 10,000 test images) using Colab's T4 GPU. Here's how they stacked up:

**CNN**:
- Accuracy: 98.76%
- F1 Score: 0.9876
- Loss: 0.0412
- Time: 45.32s

**Faster R-CNN**:
- Accuracy: 97.83%
- F1 Score: 0.9781
- Loss: 0.0698
- Time: 112.45s

**VGG16**:
- Accuracy: 99.21%
- F1 Score: 0.9920
- Loss: 0.0289
- Time: 187.63s

**AlexNet**:
- Accuracy: 98.94%
- F1 Score: 0.9893
- Loss: 0.0376
- Time: 156.19s

**ViT**:
- Accuracy: 98.52%
- F1 Score: 0.9851
- Loss: 0.0482
- Time: 203.76s

### Key Takeaways
- Pretrained models (VGG16, AlexNet) led the pack, thanks to transfer learning.
- CNN was the fastest and still scored high—great for simple tasks like MNIST.
- ViT showed transformer potential but took longer to train.
- Faster R-CNN lagged; it's overkill for basic classification.

## Prepared by
| Avatar | Name | GitHub |
|--------|------|--------|
| <img src="https://github.com/zachary013.png" width="50" height="50" style="border-radius: 50%"/> | Zakariae Azarkan | @zachary013 |

## 🤝 Contributing
This is my uni work, so I'm not accepting pull requests—but feel free to fork and experiment! Got ideas? Drop them in GitHub Issues.

## 📜 License
Licensed under the MIT License—use it, share it, just don't blame me if your GPU overheats! 🔥

## 🙌 Shoutouts
- Prof. Lotfi EL AACHAK for guiding this lab.
- PyTorch team for an awesome framework.
- Colab for free GPU access—lifesaver!
- MNIST Dataset for being the perfect playground for vision tasks.
