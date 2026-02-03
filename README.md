# 🧠 Brain Tumor MRI Classification with Grad-CAM

An AI-powered medical image analysis system that classifies brain MRI scans into tumor types and visualizes **where the model focuses** using **Grad-CAM** for explainability.

This project combines Deep Learning, Transfer Learning, Data Augmentation, and Explainable AI into an interactive **Streamlit medical app**.

---

## 🧠 Dataset

We used the **Brain Tumor MRI Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains labeled MRI scans in the following classes:

- *glioma*
- *meningioma*
- *pituitary*
- *no tumor*

Images are organized into a training and test split.

---

## 🧠 Model Backbone & Workflow

The system uses **transfer learning** from a pre-trained ResNet50 model and trains a custom classifier head + fine-tunes conv layers.

### Architecture Highlights

- Pre-trained **ResNet50** base
- Global Average Pooling
- Dense layers with dropout + normalization
- Softmax classification

Grad-CAM is used to visualize areas in the MRI scan most important for decision-making.

---

## 🚀 Features

- Classifies MRI scans into 4 classes
- Heavy data augmentation to improve generalization
- Two-phase training (head + fine-tuning)
- Class imbalance addressed with class weights
- Confusion matrix & classification report
- Grad-CAM visualization
- Interactive **Streamlit web app**

---

## 🛠️ Tech Stack

- TensorFlow / Keras
- OpenCV
- NumPy / Matplotlib / Seaborn
- Scikit-learn
- Streamlit

---

## 📁 Project Structure

.
├── Medical Brain Tumor.ipynb      # Training, evaluation, Grad-CAM
├── brain_tumor_resnet50_with_preprocessing.keras
├── app.py                        # Streamlit app with Grad-CAM
└── README.md

---

## ⚙️ Training Pipeline (Notebook)

1. Data exploration and visualization
2. Data augmentation with ImageDataGenerator
3. Compute class weights for imbalance
4. Build ResNet50 transfer learning model
5. Phase 1: Train classifier head
6. Phase 2: Fine-tune conv5 block
7. Evaluation on test set
8. Save model with preprocessing
9. Implement Grad-CAM visualization

---

## ▶️ Run the Streamlit App

Install dependencies:

pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn streamlit

Run the app:

streamlit run app.py

Upload an MRI image and see:
- Predicted tumor type
- Confidence
- Class probabilities
- Grad-CAM attention map

---

## 📊 Example Output

- Prediction: **GLIOMA**
- Confidence: **97.3%**
- Visual heatmap showing tumor region focus

---

## 🧪 Explainable AI (Grad-CAM)

Grad-CAM highlights the regions of the MRI that most influenced the model’s decision, increasing trust for medical usage.

---

## 🎯 Use Cases

- Medical imaging assistance
- Radiology AI support tool
- Educational demonstration of Explainable AI
- Deep learning in healthcare projects

---

## 👨‍💻 Author

Abdelrhman Ashraf  
Faculty of Engineering — Computers & Systems  
AI & Machine Learning Engineer
