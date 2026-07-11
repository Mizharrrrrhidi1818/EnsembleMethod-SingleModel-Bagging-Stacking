# 🚗 Vehicle Silhouette Classification using Machine Learning & Ensemble Methods (Single, Bagging and Stacking)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview
This project focuses on building highly accurate machine learning models to classify vehicles (Bus, Car, Van) based on their geometric silhouette features. By leveraging advanced ensemble learning techniques—specifically **Bagging** and **Stacking**—the project aims to maximize predictive performance for applications in smart city traffic monitoring and automated toll systems.

## 🎯 Objectives
- Clean and preprocess the Vehicle Silhouettes dataset.
- Establish a strong baseline using a single Random Forest classifier.
- Implement and evaluate Bagging and Stacking ensemble techniques.
- Compare models using Accuracy, Precision, Recall, and F1-Score.

## 📊 Dataset
The dataset is sourced from the UCI Machine Learning Repository / Kaggle. 
- **Instances:** 846
- **Features:** 18 numerical geometric features (e.g., compactness, circularity, scatter_ratio).
- **Target:** 3 classes (`bus`, `car`, `van`).

*Note: Due to GitHub file size limits, the raw dataset is not included in this repo. You can download it from [Kaggle](https://www.kaggle.com/datasets) or use the provided download script.*

## 🛠️ Methodology
1. **Data Preprocessing:** Handled missing values via row deletion, applied `StandardScaler` for feature normalization.
2. **Data Splitting:** 70/30 train-test split with stratification to maintain class balance.
3. **Modeling:**
   - **Baseline:** Random Forest Classifier.
   - **Bagging:** Ensemble of decision trees with varying bag sizes (5 to 50).
   - **Stacking:** Meta-learner approach combining Random Forest, LDA, KNN, and Gaussian Naive Bayes as base estimators, fed into a final Random Forest meta-model.

## 📈 Results & Discussion

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 94.67% | 94.67% | 94.67% | 94.67% |
| **Bagging (Best Config)** | 94.10% | 94.15% | 94.10% | 94.08% |
| **Stacking Ensemble** 🏆 | **98.78%** | **98.83%** | **98.78%** | **98.79%** |

**Key Takeaway:** While the baseline Random Forest performed exceptionally well, the **Stacking Ensemble** significantly outperformed all other models by capturing complex, non-linear relationships through diverse base learners.

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.9+ installed.


