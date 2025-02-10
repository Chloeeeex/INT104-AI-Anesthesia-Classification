# Patient Anesthesia Classification using Machine Learning

**A machine learning-based approach to recommend appropriate anesthesia methods based on past patient data.**

## Table of Contents

- [Introduction](#introduction)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
  - [Data Cleaning](#data-cleaning)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Machine Learning Models](#machine-learning-models)
  - [Supervised Classification](#supervised-classification)
  - [Unsupervised Clustering](#unsupervised-clustering)
- [Model Evaluation](#model-evaluation)
- [Results and Findings](#results-and-findings)

## Introduction

This project implements machine learning techniques to classify patients into different anesthesia recommendation groups based on their medical data. The classification helps optimize the anesthesia selection process by analyzing key features from a patient questionnaire. The dataset consists of over 5000 patient records with 15 features, and the project follows a structured pipeline:

1. **Data Preprocessing**: Cleaning and feature selection.
2. **Dimensionality Reduction**: Applying **Principal Component Analysis (PCA)**.
3. **Supervised Classification**: Implementing classifiers like **SVM, Decision Tree, Random Forest, and Naive Bayes**.
4. **Unsupervised Clustering**: Using **K-Means** to cluster patients.
5. **Model Evaluation**: Assessing accuracy using cross-validation and learning curves.

## Dataset and Preprocessing

### Data Cleaning
- The dataset includes labeled data (`0`, `1`), with some outliers labeled as `2`.
- Outliers (`label = 2`) are removed, reducing the dataset to **5330 rows**.
- Checked data distribution and correlation before applying machine learning models.
![Sample distribution](https://github.com/user-attachments/assets/dba2ad00-02eb-4ebb-978e-305239dc6850)


### Dimensionality Reduction
- **PCA (Principal Component Analysis)** was used to retain **90% of variance**, reducing complexity and improving model efficiency.

![Cumulative sum of explained variance ratio](https://github.com/user-attachments/assets/7de32b78-1d1d-4def-bb17-88fa106a8cc8)

## Machine Learning Models

### Supervised Classification
Implemented the following classifiers:
- **Support Vector Machine (SVM)**: Selected as the best-performing model.
- **Decision Tree**: Overfitted on training data, resulting in lower generalization.
- **Random Forest**: Showed stable accuracy but was computationally expensive.
- **Naive Bayes**: Fast training, but lower accuracy compared to SVM.

![Confusion Matrix of SVM classifier|200](https://github.com/user-attachments/assets/38b45542-b39c-4132-8df2-952beae1d36a)![Learning curve for classifierSVM|200](https://github.com/user-attachments/assets/2db94ae8-8fa9-41b8-ac17-43f352937299)



### Unsupervised Clustering
- **K-Means Clustering** was used to classify patients into two groups.
- The best number of clusters was determined using the **silhouette coefficient** by elbow method.
- However, clustering results were not ideal due to dataset characteristics.

![Result of K-Means1|200](https://github.com/user-attachments/assets/4c7cc49b-5100-4188-83ed-af1f7cf07252)


## Model Evaluation
- **Cross-validation scores** were used to compare models:
  
| Classifier       | Training Accuracy | Testing Accuracy |
|-----------------|------------------|-----------------|
| **SVM**         | 75%              | 69%             |
| **Decision Tree** | 88%              | 62%             |
| **Random Forest** | 88%              | 65%             |
| **Naive Bayes**  | 72%              | 67%             |

- **SVM was chosen as the final model** due to its best performance in terms of accuracy and stability.

## Results and Findings
- The **SVM classifier** achieved the best results, making it the preferred model for anesthesia recommendation.
- **K-Means clustering** was attempted but was less effective due to the dataset structure.
- Future improvements could involve **deep learning techniques** or **ensemble models** for better accuracy.
