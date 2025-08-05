# -Image-Recognition-System - Developed an image classification model to identify objects in images.

A simple image classification web app built using **TensorFlow**, **Flask**, and **OpenCV**. This project uses a Convolutional Neural Network (CNN) trained on the **CIFAR-10 dataset** to classify images into 10 categories.

---

## 📁 Project Structure
image_recognition_app/
├── app.py # Flask web application
├── train_model.py # CNN training script (CIFAR-10)
├── model/
│ └── model.h5 # Saved trained model
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Result page
├── static/
│ └── style.css # Styling (optional)
├── uploads/ # Uploaded images
└── requirements.txt # Project dependencies

 ## Install Dependencies
pip install -r requirements.txt

## Run the App
python app.py

🧠 Model Info
Dataset: CIFAR-10

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Framework: TensorFlow + Keras

📜 Requirements
Python 3.10+

TensorFlow

Flask

NumPy

Pillow

 Author
Sourav Paul
Intern @ Codec Technologies
Email:- souravpaul043@gmail.com


