# plant-disease-classification-efficientnet
A deep learning project that uses EfficientNetB0 and TensorFlow to classify plant leaf images into 38 disease and healthy categories. Includes a Streamlit web application for image upload and disease prediction.



## 📌 About the Project

This project uses deep learning to identify plant leaf diseases from images. It uses a pretrained EfficientNetB0 model with TensorFlow and Keras to classify leaf images into 38 plant disease and healthy categories.

A Streamlit web application allows users to upload a leaf image and view the predicted category along with its confidence score.

## ✨ Features

- Classifies images into 38 plant categories.
- Uses pretrained EfficientNetB0 for feature extraction.
- Provides a simple web interface using Streamlit.
- Displays the predicted category and confidence score.
- Supports JPG, JPEG, and PNG images.
- Includes a separate Python script for terminal-based predictions.
- Supports model inspection and training-result tracking with Weights & Biases.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- EfficientNetB0
- NumPy
- Pillow (PIL)
- Streamlit
- Weights & Biases (W&B)

## 🧠 Model Details

- **Architecture:** EfficientNetB0
- **Input image size:** 128 × 128 pixels
- **Input channels:** 3 (RGB)
- **Output categories:** 38
- **Pretrained weights:** ImageNet
- **Training approach:** Frozen pretrained base with a trainable classification head

## 📂 Project Structure

```text
plant-disease-classification-efficientnet/
├── app.py
├── predict.py
├── new.py
├── og_to_wandb.py
├── plant_disease_model.keras
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/plant-disease-classification-efficientnet.git
cd plant-disease-classification-efficientnet
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Run the Web Application

```bash
streamlit run app.py
```

Upload a plant leaf image, click **Predict Disease**, and view the prediction result.

## 🔍 Run Prediction in the Terminal

```bash
python predict.py
```

Enter the path to the image you want to test when prompted.

## 📊 Model Inspection

To display the model architecture and parameter count, run:

```bash
python new.py
```

## 📈 Training Results

The project includes a script for logging recorded training and validation metrics to Weights & Biases.

```bash
python og_to_wandb.py
```

This script requires W&B setup and the saved model file. It logs the metrics recorded in the script rather than retraining the model.

## ⚠️ Limitations

- Predictions depend on image quality and how well the input matches the training data.
- The confidence score is the model's predicted probability, not a guarantee of diagnostic correctness.
- The application is intended as a machine learning project and should not replace professional agricultural diagnosis.

## 👨‍💻 Author

**YOUR NAME**

## 📄 License

Add a license if you intend to distribute the project for others to use.
