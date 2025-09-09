🖼️ Image Recognition System

A simple image classification web app built using TensorFlow, Flask, and OpenCV.
This project uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 categories.

📂 Project Structure
image_recognition_app/
├── app.py              # Flask web application
├── train_model.py      # CNN training script (CIFAR-10)
├── model/
│   └── model.h5        # Saved trained model
├── templates/
│   ├── index.html      # Upload page
│   └── result.html     # Result page
├── static/
│   └── style.css       # Styling (optional)
├── uploads/            # Uploaded images
└── requirements.txt    # Project dependencies

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/image-recognition-system.git
cd image-recognition-system


Install dependencies:

pip install -r requirements.txt


Run the Flask app:

python app.py


Open in your browser:

http://127.0.0.1:5000/

🧠 Model Information

Dataset: CIFAR-10

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Framework: TensorFlow + Keras

Model: CNN trained and saved as model/model.h5

📜 Requirements

Python 3.10+

TensorFlow

Flask

NumPy

Pillow

OpenCV

A requirements.txt for your Image Recognition System project (TensorFlow + Flask + OpenCV + CIFAR-10 CNN).

Flask==2.3.3
tensorflow==2.15.0
keras==2.15.0
opencv-python==4.8.0.74
numpy==1.24.3
Pillow==10.0.0
gunicorn==21.2.0

🔑 Why these packages?

Flask → Web framework for the app.

TensorFlow + Keras → CNN training & inference.

OpenCV → Image preprocessing & handling uploads.

NumPy → Array operations for image data.

Pillow → Image file processing (JPG, PNG).

Gunicorn → For deployment (Heroku, AWS, etc.).

👨‍💻 Author

Sourav Paul
Intern @ Codec Technologies
📧 Email: souravpaul043@gmail.com
