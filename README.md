# cyber-attack-detection-ml

# Machine Learning Techniques for Cyber Attacks Detection

This project focuses on detecting cyber attacks using Machine Learning algorithms.  
The system preprocesses network-related data, applies multiple ML models, evaluates their performance, and deploys the best-performing model using a Flask web application.

## 🚀 Project Objectives

- Analyze cyber attack data using Exploratory Data Analysis (EDA)
- Train multiple Machine Learning models
- Compare accuracy and performance
- Select the best model for deployment
- Build a Flask-based web application for real-time prediction

## 🧠 Machine Learning Algorithms Used

- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)

> **Conclusion:** Decision Tree and Random Forest models provided the highest accuracy.


## 🛠️ Technologies Used

- **Programming Language:** Python 3.7
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Scikit-learn  
  - Flask
- **Tools:** Jupyter Notebook, VS Code

## 📂 Project Structure

cyber-attack-detection-ml/
├── README.md
├── requirements.txt
├── train_model.ipynb
├── model_training.py
├── app.py
├── model.pkl
├── scaler.pkl
├── dataset/
│ └── cyber_data.csv
├── templates/
│ └── index.html
└── static/
└── style.css

## ⚙️ How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model_training.py
```
This will generate:

model.pkl
scaler.pkl

### 3. Run the Flask Application 
```bash
python app.py
```

### 4. Open in Browser
```bash
http://127.0.0.1:5000/
```

📊 Output

User enters network parameters

System predicts whether the input represents a Cyber Attack or Normal Traffic

