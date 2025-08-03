# ML_Classifier
This project builds a simple Logistic Regression classifier for the Iris dataset with a focus on testability. It includes modular code for data preprocessing, model training, and evaluation, along with unit and integration tests using pytest to ensure pipeline correctness and model accuracy.

# 🤖 ML_Classifier

A modular and extensible machine learning classifier pipeline built with scikit-learn and Python. This project allows you to train, evaluate, and test multiple ML models on structured datasets for binary or multi-class classification tasks.

---

## 📌 Project Description

**ML_Classifier** is a complete classification framework that supports data preprocessing, feature selection, model training, hyperparameter tuning, and evaluation. It includes implementations of popular classifiers like Logistic Regression, SVM, Random Forest, KNN, and simple Neural Networks. The project is designed for quick experimentation and benchmarking on classification problems across domains.

---

## 🚀 Features

- 🧼 Data preprocessing and cleaning
- 🔍 Feature selection and scaling
- 🤖 Multiple ML models (Logistic Regression, SVM, Random Forest, etc.)
- ⚙️ Hyperparameter tuning with GridSearchCV
- 📊 Model evaluation (accuracy, precision, recall, F1, ROC-AUC)
- 📁 Support for CSV input and result logging

---

## 📂 Project Structure

ML_Classifier/
├── data/
│ └── dataset.csv
├── models/
│ ├── random_forest.py
│ ├── svm.py
│ └── neural_net.py
├── utils/
│ ├── preprocess.py
│ └── metrics.py
├── main.py
├── requirements.txt
└── README.md


---

🧠 Usage
  python main.py --model random_forest --input data/dataset.csv
You can customize the model and dataset path using command-line arguments.

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/ML_Classifier.git
cd ML_Classifier
pip install -r requirements.txt


🧪 Evaluation Metrics
 - Accuracy
 - Precision / Recall
 - F1-Score
 - ROC-AUC
 - Confusion Matrix

🛠️ Technologies Used
 - Python
 - scikit-learn
 - pandas
 - matplotlib / seaborn
 - NumPy
