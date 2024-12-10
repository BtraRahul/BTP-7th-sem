Here’s a well-structured **README** file template for your GitHub repository that clearly explains your **Brain Tumor Classification using Deep Learning** project.

---

# **Brain Tumor Classification Using Deep Learning**

This project aims to classify brain tumors into four categories (**Glioma, Meningioma, Pituitary Tumor, and Normal**) using **ResNet-101** deep learning architecture and MRI images.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Model Training](#model-training)
6. [Performance Metrics](#performance-metrics)
7. [Results](#results)
8. [How to Run the Project](#how-to-run-the-project)
9. [Future Scope](#future-scope)
10. [Contributors](#contributors)

---

## **Project Overview**
Brain tumors are a critical health issue, and early detection is vital for treatment. This project uses deep learning techniques, specifically **ResNet-101** with transfer learning, to classify brain tumors from MRI images into:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **Normal (No Tumor)**

The project demonstrates the preprocessing pipeline, model training, evaluation, and a simple **Graphical User Interface (GUI)** for predictions.

---

## **Dataset**
- Source: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets)
- **Classes**:  
   - Glioma Tumor  
   - Meningioma Tumor  
   - Pituitary Tumor  
   - Normal  
- **Total Images**: 3,264 images  
- Data Preprocessing:
   - Resized images to **224x224 pixels**.  
   - Applied **data augmentation** (rotation, zoom, shear, flips) to improve generalization.  
   - Normalized pixel values using ResNet's `preprocess_input` function.  

---

## **Technologies Used**
- **Programming Language**: Python 3.x  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Model Architecture**: ResNet-101 (Transfer Learning)  
- **Data Visualization**: Matplotlib  
- **Libraries**:  
   - `NumPy`  
   - `Pandas`  
   - `scikit-learn`  
   - `matplotlib`  
   - `tensorflow`  

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/BtraRahul/BTP-7th-sem.git
   cd BTP-7th-sem/code
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is in the proper directory structure:
   ```
   brain-tumor-dataset/
      glioma_tumor/
      meningioma_tumor/
      pituitary_tumor/
      no_tumor/
   ```

---

## **Model Training**
- Used **ResNet-101** pre-trained on ImageNet with custom top layers for 4-class classification.  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Batch Size**: 32  
- **Epochs**: 15  

Run the following script to train the model:
```bash
python resnet101.py
```

The trained model will be saved as `resnet101_trained_model.h5`.

---

## **Performance Metrics**
The model performance was evaluated using the following metrics:
1. **Accuracy**  
2. **Precision**  
3. **Recall**  
4. **F1-Score**  
5. **Confusion Matrix**  

---

## **Results**
- **Validation Accuracy**: 88.37%  
- **ResNet-101** performed better than ResNet-50 due to its deeper architecture, which captures more intricate features.  

**Sample Predictions**:
- Glioma Tumor → **Detected**  
- Meningioma Tumor → **Detected**  
- No Tumor → **Detected**

---

## **How to Run the Project**
1. **Train the Model**:  
   Run the `train_model.py` script to train the ResNet-101 model on the MRI dataset.

2. **Prediction**:
   - A simple GUI can be added to upload an MRI image and get predictions.
   - You can load the trained model for predictions:
     ```python
     from tensorflow.keras.models import load_model
     import numpy as np
     from tensorflow.keras.preprocessing import image

     model = load_model('resnet101_trained_model.h5')
     img_path = 'path/to/image.jpg'
     img = image.load_img(img_path, target_size=(224, 224))
     img_array = image.img_to_array(img)
     img_array = np.expand_dims(img_array, axis=0)
     img_array /= 255.0  # Normalization
     predictions = model.predict(img_array)
     print(predictions)
     ```

3. **Run GUI** (Optional):  
   Add a Python GUI using tools like `tkinter` or `streamlit`.

---

## **Future Scope**
1. Add more tumor classes for classification.
2. Use more advanced architectures like EfficientNet or Vision Transformers.
3. Deploy the model as a **web application** or mobile app for real-time predictions.
4. Incorporate explainable AI techniques to visualize why the model made certain predictions.

---

## **Contributors**
- **Rahul Batra** (2021UIT3140)  
- **Aditi Sah** (2021UIT3077)  
- **Pranav Singh Kanwar** (2021UIT3135)  

---

## **Acknowledgments**
- Special thanks to **Dr. Nisha Kandhoul** for guidance and mentorship.  
- Dataset sourced from **Kaggle**.  

---

Feel free to customize it further to match your needs! 🚀
