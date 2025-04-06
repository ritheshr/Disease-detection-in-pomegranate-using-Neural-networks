# 🍎 Pomegranate Disease Detection using Machine Learning & Deep Learning
A smart agriculture project that uses machine learning and deep learning techniques to detect common pomegranate diseases such as fruit rot, scab, and bacterial blight. This system enables farmers to upload fruit images and receive instant feedback on the health status of the fruit, helping prevent crop loss and improving yield.

# 🚀 Project Overview
This project aims to assist farmers in early identification of diseases in pomegranate fruits using image classification techniques. It combines traditional machine learning methods (SVM + GLCM features) and deep learning models (CNNs) to deliver accurate and fast disease predictions.

# 📌 Key Features
📷 Image-Based Detection: Uses image processing and feature extraction to identify lesions and disease symptoms.

# 🧠 Dual-Model Approach:

Support Vector Machine (SVM) with GLCM texture features.

Convolutional Neural Network (CNN) for improved classification.

# 🎯 83% Accuracy: Achieved reliable accuracy on a dataset of 400 images.

# 🌐 Web Interface: Built a user-friendly web app where farmers can upload fruit images to receive instant predictions.

# 💡 Smart Agriculture: Supports sustainable farming by reducing reliance on manual inspection and minimizing losses.

# 🧪 Dataset
Total Images: 400

150 - Fruit Rot

90 - Scab

160 - Healthy Fruits

Images were preprocessed using normalization, resizing, and segmentation to enhance model training.

# 🧰 Technologies Used
Python

OpenCV – Image preprocessing

scikit-learn – SVM modeling

TensorFlow / Keras – CNN development

GLCM (Gray-Level Co-occurrence Matrix) – Feature extraction

Flask / Streamlit – Web app (for deployment)

NumPy, Matplotlib – Data manipulation and visualization

# 📈 Workflow
Image Preprocessing:

Resize images

Normalize pixel values

Segment lesions

Feature Extraction:

Use GLCM to extract texture-based features

Model Training:

SVM for traditional classification

CNN for improved accuracy

Model Evaluation:

Accuracy: ~83%

Confusion matrix and precision-recall metrics

Deployment:

Flask or Streamlit-based web interface

Accepts image input and displays disease prediction


# Create virtual environment and activate
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt


# 📸 Screenshots
(Add images of the app interface and sample predictions here)

# 🔍 Results
SVM with GLCM Features: 83% accuracy

CNN Model: Improved feature extraction and classification precision

Real-time detection interface for practical agricultural usage

# 🎯 Future Enhancements
Expand dataset to include more diseases and variations

Integrate multilingual support for farmers across regions

Deploy on mobile platforms for easy access

Add explainable AI components for transparency

# 🤝 Contributing
Feel free to fork the project and open issues or PRs for improvements!
