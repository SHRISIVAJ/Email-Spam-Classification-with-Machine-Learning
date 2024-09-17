#Email Spam Classification with Machine Learning
This project implements a Naive Bayes classifier to detect spam emails using word frequency features. The model is trained and tested on a dataset where the frequency of certain words in emails is used as features to classify them as spam or non-spam.

Project Overview
The primary objective of this project is to classify emails as spam or not spam using machine learning techniques. The dataset contains several email samples, each represented as numerical features extracted from the text. The final model achieves an accuracy of 94.85%, indicating high performance for identifying spam emails.

Key Features:
Dataset: Contains word frequency counts in emails.
Model: Multinomial Naive Bayes.
Metrics: Accuracy, precision, recall, F1-score, and confusion matrix.
Table of Contents
Installation
Dataset
Model
Usage
Results
Contributing
License
Installation
To run this project locally, you need to have Python 3.x installed, along with the following Python libraries:

pandas
scikit-learn
numpy
You can install the dependencies using pip:

pip install pandas scikit-learn numpy


Optional
You can also run this project in Google Colab by uploading the dataset and executing the provided code in a notebook.


Dataset
The dataset used for this project is a CSV file containing the following:

Word frequency features: A set of numerical columns representing the occurrence of specific words in each email.
Prediction label: A binary label (0 or 1), where 0 indicates non-spam (ham) and 1 indicates spam.

Model
This project uses the Naive Bayes algorithm, specifically the Multinomial Naive Bayes, which is well-suited for classification tasks involving word frequency counts.

Steps:

Data Preprocessing: Cleaning and preparing the data, including handling missing values and splitting it into training and testing sets.
Feature Extraction: Using word frequency features from the dataset to represent the emails numerically.
Model Training: Training the Naive Bayes model on the training dataset.
Evaluation: Using metrics such as accuracy, precision, recall, F1-score, and confusion matrix to evaluate the model performance.

Usage
1. Clone this repository:

git clone https://github.com/your-username/spam-email-classification.git
cd spam-email-classification

2. Make sure the required Python libraries are installed.

3. Run the project (on Colab or locally):
python main.py

Alternatively, if you're using Google Colab, upload the CSV file and run the code cells provided in the notebook.
