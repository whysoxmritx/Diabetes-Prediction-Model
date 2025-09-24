# 🩺 Diabetes Prediction using SVM  

This project implements a **Machine Learning model** that predicts whether a person is diabetic or not based on medical diagnostic measurements from the **PIMA Indian Diabetes Dataset**. The model is built using **Support Vector Machine (SVM) with a linear kernel**, and data preprocessing steps such as **standardization** are applied for better performance.  

---

## 🔑 Key Features  
- Uses the **PIMA Indian Diabetes Dataset** (`diabetes.csv`).  
- Preprocesses the data with **StandardScaler** to normalize features.  
- Splits dataset into **training (80%)** and **testing (20%)** sets with stratified sampling.  
- Trains an **SVM classifier** with a linear kernel.  
- Evaluates model performance with **accuracy score** on both training and test data.  
- Includes a **custom prediction system** where user input can be tested against the trained model.  

---

## ⚙️ Tech Stack  
- **Python**  
- **NumPy**  
- **Pandas**  
- **Scikit-learn**  

---

## 📊 Workflow  
1. Load the dataset (`diabetes.csv`).  
2. Explore and preprocess the data (feature scaling).  
3. Train the SVM classifier.  
4. Evaluate accuracy on training and test data.  
5. Build a prediction system to classify new input data.  
