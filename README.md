# 🐱🐶 Cats vs Dogs Image Classifier using PyTorch

A Deep Learning image classification project built with **PyTorch** that classifies images as either **Cat** or **Dog** using a Convolutional Neural Network (CNN).

---

## 📌 Project Overview

This project demonstrates how to build, train, evaluate, save, and use a CNN model for binary image classification.

The model is trained on a Cats vs Dogs dataset and can predict whether a new image contains a **cat** or a **dog**.

---

## 📂 Project Structure

```
PETVISION/
│
├── Dataset/                  # Training and Testing Dataset
│
├── saved_models/             # Saved trained model (.pth)
│   └── image_classifier.pth
│
├── img1.png                  # Sample prediction image
├── img2.png
├── img3.png
│
├── main.ipynb                # Main Jupyter Notebook
├── requirements.txt          # Required Python packages
├── README.md
└── .gitignore
```

---

## 🚀 Features

- Image preprocessing using `torchvision.transforms`
- Custom CNN model
- Training and validation pipeline
- Accuracy and loss tracking
- Save and load trained model
- Predict new images
- GPU support (CUDA)

---

## 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)
- tqdm

---

## 📥 Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

Move into the project folder

```bash
cd your-repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Structure

```
Dataset/
│
├── train/
│   ├── Cat/
│   └── Dog/
│
└── test/
    ├── Cat/
    └── Dog/
```

---

## 🧠 Model Architecture

The CNN consists of:

- Conv2D
- ReLU
- Batch Normalization
- MaxPooling
- Flatten
- Fully Connected (Linear) Layer

Example Architecture:

```
Input Image (3×224×224)

↓

Conv Block 1
(3 → 32)

↓

Conv Block 2
(32 → 64)

↓

Conv Block 3
(64 → 128)

↓

Conv Block 4
(128 → 256)

↓

Conv Block 5
(256 → 512)

↓

Flatten

↓

Linear Layer

↓

Output
Cat / Dog
```

---

## ▶️ Train the Model

Run the notebook

```
main.ipynb
```

or execute the training cells.

The trained model will be saved inside

```
saved_models/image_classifier.pth
```

---

## 🔍 Predict on a New Image

Load the trained model

```python
model.load_state_dict(
    torch.load("saved_models/image_classifier.pth", map_location=device)
)

model.eval()
```

Load an image

```python
image = Image.open("img1.png")
```

Apply transforms

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
```

Predict

```python
prediction = model(image.unsqueeze(0))
```

---

## 📈 Training Results

The notebook records

- Training Loss
- Training Accuracy
- Testing Loss
- Testing Accuracy

These metrics can be plotted using Matplotlib.

---

## 📷 Sample Images

Example prediction images included:

- img1.png
- img2.png
- img3.png

---

## 📦 Requirements

Install all required packages

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- Transfer Learning (ResNet18, EfficientNet)
- Streamlit Web App
- Flask API
- Multi-class Image Classification
- Real-time Webcam Prediction
- Model Deployment

---

## 👨‍💻 Author

**Debjit Das**

B.Tech CSE (AI & ML)

Passionate about Artificial Intelligence, Deep Learning, Computer Vision, and PyTorch.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub.