<h1>Fake Job Post Prediction</h1>

<h3>Project Overview</h3>

This project focuses on detecting fake job postings using machine learning techniques. It involves data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation to ensure high accuracy and reliability. By analyzing job descriptions and related features, the model aims to distinguish between real and fraudulent job listings.

<h3>Dataset</h3>
-The dataset used for this project is available on Kaggle:<br>
 https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction<br>
- Contains job postings labeled as real or fake.<br>
- Includes diverse textual and numerical features such as job title, company profile, job description, requirements, and benefits.<br>
- Imbalanced dataset, requiring strategies for handling class distribution.<br>

<h3>Data Preprocessing</h3>

- Handled missing values by imputing or dropping irrelevant columns.<br>
- Tokenized and vectorized textual features using TF-IDF.<br>
- Applied label encoding for categorical variables.<br>
- Scaled numerical features for improved model performance.<br>

<h3>Exploratory Data Analysis (EDA)</h3>

- Analyzed class distribution and applied SMOTE for class balancing.<br>
- Visualized word clouds and feature importance.<br>
- Identified correlations and feature interactions.<br>

<h3>Model Training</h3>

<h3>Models Implemented</h3>

- Logistic Regression<br>
- Decision Trees<br>
- Random Forest<br>

<h3>Handling Class Imbalance</h3>

- Implemented SMOTE(Synthetic Minority Over-sampling Technique) to balance the dataset.<br>
- Optimized class weights to enhance precision and recall.<br>

<h3>Hyperparameter Tuning</h3>

- Utilized RandomizedSearchCV for optimizing model parameters.<br>
- Evaluated models based on F1-score, precision, recall, and accuracy.<br>

<h3>Model Evaluation</h3>

- Used confusion matrix, classification report, and accuracy score.<br>
- Optimized for better recall and precision to effectively detect fake job posts.<br>

<h3>Best Model Performance</h3>

- Precision: 74%<br>
- Recall: 80%<br>
- F1-Score: 77%<br>
- Accuracy: 97.67%<br>

<h3>Future Enhancements</h3>

- Implement deep learning models (LSTMs, Transformers) for advanced text processing.<br>
- Explore BERT embeddings, Word2Vec for NLP-based feature engineering.<br>
- Integrate anomaly detection for improved fraud detection.<br>
- Deploy as a web application for real-time job post classification.<br>



