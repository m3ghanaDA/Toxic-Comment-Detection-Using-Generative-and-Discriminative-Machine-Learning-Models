# Toxic Comment Detection in Online Forums

## 1. Problem Definition
### Objective
Develop machine learning models to automatically classify comments as **toxic** or **non-toxic**, ensuring compliance with community guidelines and improving user experience.

### Task
Binary classification of comments for toxicity detection.

---

## 2. Data Collection and Understanding
### Dataset
- **Training Set**: 8699 comments (13% toxic, 87% non-toxic)
- **Validation Set**: 2920 comments (13.15% toxic, 86.85% non-toxic)
- **Test Set**: 2896 comments (labels unavailable for testing).

### Key Challenge
Significant class imbalance: Majority of comments are non-toxic.

---

## 3. Data Preprocessing
### Steps
1. **Text Normalization**:
   - Converted text to lowercase.
   - Removed stopwords, punctuation, and special characters.
   - Performed lemmatization to reduce words to their root forms.
   - Filtered non-ASCII characters and removed short words (< 3 characters).

2. **Class Imbalance Handling**:
   - Undersampled the majority class (non-toxic) by 35%.
   - Final distribution: 81.5% non-toxic, 18.5% toxic.

---

## 4. Exploratory Data Analysis (EDA)
### Insights
- **Comment Length**: Most comments are short, with few exceeding 100 words.
- **Class Imbalance**: Visualized distribution confirmed high imbalance.
- **Common Toxic Words**: Word clouds and frequency charts highlighted common toxic terms.

---

## 5. Feature Engineering
### Methods
- **Bag-of-Words (BoW)**: Used for numerical feature representation.
- **TF-IDF**: Tested but did not outperform BoW, so BoW was selected for simplicity and effectiveness.

---

## 6. Model Selection and Training
### Models
1. **Naive Bayes** (Generative):
   - Laplace smoothing (alpha = 0.1) applied.
   - Optimized class prior probabilities.

2. **Logistic Regression** (Discriminative):
   - Hyperparameter tuning with solvers, class weights, and optimizers.

3. **Support Vector Machine (SVM)**:
   - Linear kernel with tuned parameters: regularization (C=5), class weights, and intercept scaling.

### Optimization
- **GridSearchCV**: Used 5-fold cross-validation, optimizing for F1-score to balance precision and recall.

---

## 7. Model Evaluation
### Metrics
- **Accuracy**: Overall correct predictions.
- **F1-Score**: Balances precision and recall for imbalanced datasets.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.

### Results
| Model                | Accuracy | F1-Score | Precision | Recall  |
|----------------------|----------|----------|-----------|---------|
| **Naive Bayes**      | Moderate | Lower    | Lower     | Lower   |
| **Logistic Regression** | Similar to NB | Moderate | Slightly lower recall |
| **SVM**              | **Highest** | **Best Balance** | **Best Performance** |

- **SVM**: Outperformed other models with the highest F1-score, precision, and recall, making it the best-performing model.

---

## 8. Prediction and Analysis
### Test Set Predictions
- **Naive Bayes**: Predicted 619 toxic comments.
- **SVM**: Predicted 1005 toxic comments.

### Observations
- SVM outperformed Naive Bayes in capturing subtle nuances of toxicity.
- Some false negatives occurred due to nuanced or implicit toxic language.

---

## 9. Conclusions and Insights
- **Best Model**: Support Vector Machine (SVM) due to its superior performance in balancing precision and recall.
- **Challenges**: 
  - Class imbalance posed a significant challenge, partly mitigated by undersampling.
  - Subtle nuances in toxic language require advanced modeling for full capture.

---

## 10. Future Work
1. **Advanced Text Representations**:
   - Implement word embeddings (e.g., Word2Vec, GloVe).
   - Explore contextual models like BERT or RoBERTa.

2. **Ensemble Methods**:
   - Use stacking or boosting to combine model strengths.

3. **Data Augmentation**:
   - Apply oversampling (e.g., SMOTE) or cost-sensitive learning.

4. **Feature Engineering**:
   - Include user metadata and conversation context for enhanced insights.

---

## 11. Lessons Learned
### Key Takeaways
- **Model Transparency**:
  - Simpler models like Naive Bayes and Logistic Regression are more interpretable but less effective for nuanced tasks.
- **Imbalance Handling**:
  - Addressing class imbalance is crucial in real-world classification problems.
- **Preprocessing Importance**:
  - Effective text preprocessing has a significant impact on model performance.

---

## Appendix
- Code and results are available upon request.
