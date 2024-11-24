1. Problem Definition
The goal of this project is to detect toxic comments in online forums automatically using machine learning models. Toxic speech on online platforms violates community guidelines and can negatively impact user experience. The task involves performing binary classification to label comments as either toxic or non-toxic, fulfilling both legal compliance and user satisfaction requirements.



2. Data Collection and Understanding
The dataset used for this project consists of three parts: training, validation, and test sets, each containing comments and toxicity labels (except the test set). The training set includes 8699 comments, with 13% toxic and 87% non-toxic, highlighting a significant class imbalance problem. The project focuses on processing text data (comments) to detect toxicity effectively.

Dataset Breakdown:

Training set: 8699 comments, 13% toxic

Validation set: 2920 comments, 13.15% toxic

Test set: 2896 comments (no toxicity labels provided)



3. Data Preprocessing
Data preprocessing is essential for preparing the raw text for machine learning models. The steps performed include:

Lowercasing the text

Removing stopwords using NLTK’s stopword list

Stripping punctuation and special characters using regular expressions

Lemmatizing the words to convert them to their root form

Filtering non-ASCII characters to clean the dataset

Removing short words (length less than 3 characters)

A key challenge addressed during preprocessing was the class imbalance, which was handled by undersampling the majority class (non-toxic comments) by removing 35% of the non-toxic rows, leading to a more balanced dataset (81.5% non-toxic, 18.5% toxic).



4. Exploratory Data Analysis (EDA)
After preprocessing, visualizations were performed to understand the data better:

Comment Length Distribution: Most comments were relatively short, with very few longer than 100 words.

Class Imbalance Visualization: Confirmed the high imbalance between toxic and non-toxic comments.

Common Words in Toxic Comments: Created a word cloud or frequency distribution of words that commonly appear in toxic comments.

This EDA helped uncover patterns in the data and informed later steps in model building and feature engineering.



5. Feature Engineering
For feature extraction, the Bag-of-Words (BoW) approach was employed, converting the text data into numerical features based on word frequencies. Although TF-IDF (Term Frequency-Inverse Document Frequency) was also tried, it did not yield better results than BoW. BoW was selected for its simplicity and effectiveness in text classification tasks.



6. Model Selection and Training
Both generative and discriminative machine learning models were explored for text classification:

Generative Model: Naive Bayes

Discriminative Models: Logistic Regression and Support Vector Machine (SVM)

Model Training and Hyperparameter Tuning:

Naive Bayes: Tuned with Laplace smoothing (alpha = 0.1) and class prior probabilities.

Logistic Regression: Tuned using GridSearchCV with different solvers, class weights (to handle imbalance), and optimization techniques.

SVM: A linear SVM was used with hyperparameters tuned for regularization (C=5), class weights, and intercept scaling.

GridSearchCV with cross-validation (5 splits) was employed for all models, optimizing based on the F1-score, which balances precision and recall, especially important in this imbalanced classification problem.



7. Model Evaluation
The models were evaluated based on the following metrics:

Accuracy

F1-score (macro)

Precision (macro)

Recall (macro)

Model Performance:

Naive Bayes: Moderate performance with reasonable accuracy but lower precision and recall.

Logistic Regression: Similar performance to Naive Bayes, with slightly lower recall.

SVM: Achieved the highest F1-score and offered the best balance between precision and recall, making it the state-of-the-art (SoTA) model for this project.



8. Prediction and Analysis
Once the models were trained, they were used to make predictions on the test set:

Naive Bayes: Predicted 619 toxic comments (out of 2896) in the test set.

SVM: Predicted 1005 toxic comments.

Example Analysis:

For some instances, both models failed to capture toxicity due to subtle language nuances.

In other cases, SVM performed better in correctly identifying aggressive or toxic comments, demonstrating superior performance.



9. Conclusions and Insights
The Support Vector Machine (SVM) emerged as the best-performing model for this task, striking a balance between precision and recall, making it well-suited for toxic comment detection in real-world scenarios. However, there were some areas for improvement:

Class imbalance posed a challenge that was addressed through undersampling, though other techniques (like SMOTE) could be explored.

Naive Bayes and Logistic Regression were more transparent but less effective than SVM.



10. Future Work
The project could be extended by exploring the following:

Advanced Text Representations: Implementing word embeddings (e.g., Word2Vec, GloVe) or contextual language models (e.g., BERT, RoBERTa) could improve model performance by capturing deeper semantic relationships.

Ensemble Methods: Techniques like stacking or boosting could combine the strengths of multiple models to improve prediction accuracy.

Data Augmentation: Handling the class imbalance using oversampling methods like SMOTE or applying cost-sensitive learning could enhance model robustness.

Feature Engineering: Additional features like user metadata or conversation context could be incorporated to better detect toxic behavior.



11. Lessons Learned
Model Transparency: While deep learning models could have been explored, simpler models like Naive Bayes and Logistic Regression were preferred for their transparency and interpretability, which is critical for identifying biases in toxic speech detection.

Imbalance Handling: Handling imbalanced datasets is key in real-world text classification problems, and experimenting with different techniques to balance classes should be prioritized.

Text Preprocessing: High-quality text preprocessing significantly impacts model performance and is a vital step in any natural language processing (NLP) project.
