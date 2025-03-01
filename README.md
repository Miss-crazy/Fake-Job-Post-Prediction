<h1>Fake Job Post Prediction</h1>

<h3>Project Overview</h3>

This project focuses on detecting fake job postings using machine learning techniques. It involves data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation to ensure high accuracy and reliability. By analyzing job descriptions and related features, the model aims to distinguish between real and fraudulent job listings.

<h3>Dataset</h3>

- Contains job postings labeled as real or fake.
- Includes diverse textual and numerical features such as job title, company profile, job description, requirements, and benefits.
- Imbalanced dataset, requiring strategies for handling class distribution.

<h3>Data Preprocessing</h3>

- Handled missing values by imputing or dropping irrelevant columns.
- Tokenized and vectorized textual features using TF-IDF.
- Applied label encoding for categorical variables.
- Scaled numerical features for improved model performance.

<h3>Exploratory Data Analysis (EDA)</h3>

- Analyzed class distribution and applied SMOTE for class balancing.
- Visualized word clouds and feature importance.
- Identified correlations and feature interactions.

<h3>Model Training</h3>

<h3>Models Implemented</h3>

- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost

<h3>Handling Class Imbalance</h3>

- Implemented SMOTE(Synthetic Minority Over-sampling Technique) to balance the dataset.
- Optimized class weights to enhance precision and recall.

<h3>Hyperparameter Tuning</h3>

- Utilized RandomizedSearchCV for optimizing model parameters.
- Evaluated models based on F1-score, precision, recall, and accuracy.

<h3>Model Evaluation</h3>

- Used confusion matrix, classification report, and accuracy score.
- Optimized for better recall and precision to effectively detect fake job posts.

<h3>Best Model Performance</h3>

- Precision: 74%
- Recall: 80%
- F1-Score: 77%
- Accuracy: 97.67%

<h3>Future Enhancements</h3>

- Implement deep learning models (LSTMs, Transformers) for advanced text processing.
- Explore BERT embeddings, Word2Vec for NLP-based feature engineering.
- Integrate anomaly detection for improved fraud detection.
- Deploy as a web application for real-time job post classification.



