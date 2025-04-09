# Clinical Trials Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.12+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

This repository contains code for an NLP-based classification system that categorizes clinical trial descriptions into different medical conditions. The system leverages both traditional ML approaches and state-of-the-art transformer models to achieve high accuracy in categorizing medical texts.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Model Deployment](#model-deployment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Google Colab Deployment](#google-colab-deployment)
- [Contributors](#contributors)
- [References](#references)

---

## Problem Statement

Clinical trial descriptions contain rich medical information but are typically unstructured. Automatically classifying these descriptions into specific medical conditions can facilitate:
- Improved searchability of clinical trial databases
- Enhanced patient-trial matching
- Better understanding of research trends across different conditions

This project aims to develop a robust classifier that categorizes clinical trial descriptions into five medical conditions: **ALS**, **Dementia**, **Obsessive Compulsive Disorder**, **Parkinson's Disease**, and **Scoliosis**.

---

## Dataset

The dataset consists of **1,759 clinical trial descriptions** with the following distribution:
- ALS: 368 trials (20.9%)
- Dementia: 368 trials (20.9%)
- Obsessive Compulsive Disorder: 358 trials (20.4%)
- Scoliosis: 335 trials (19.0%)
- Parkinson's Disease: 330 trials (18.8%)

**Dataset structure**:
- Shape: (1759, 3)
- Columns: nctid (object), description (object), label (object)
- No missing values in any column

**Description length statistics**:
- Average (mean) length: **2,016.46 characters**
- Standard deviation: **2,103.19 characters**
- Minimum length: **9 characters**
- 25th percentile: **747 characters**
- Median length: **1,406 characters** 
- 75th percentile: **2,429.5 characters**
- Maximum length: **21,765 characters**

**Word count statistics**:
- Average (mean) count: **296.64 words**
- Standard deviation: **310.79 words**
- Minimum count: **2 words**
- 25th percentile: **112 words**
- Median count: **207 words**
- 75th percentile: **355 words**
- Maximum count: **3,297 words**

The dataset is well-balanced across all five medical conditions, which is beneficial for building a robust classifier. The variance in text length suggests the need for effective text normalization techniques.

---

## Methodology

### 1. Data Exploration and Preprocessing

- **Text cleaning**: Removal of special characters, digits, and punctuation
- **Normalization**: Text lowercasing, lemmatization
- **Stopword removal**: Both standard English stopwords and domain-specific medical stopwords
- **Vocabulary analysis**: Distinctive term identification for each medical condition

### 2. Feature Engineering

Multiple feature extraction methods were evaluated:
- Binary Occurrence (Bag of Words)
- Term Frequency (TF)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- IDF Only (Inverse Document Frequency)

For transformer-based models, the raw text was tokenized using model-specific tokenizers.

### 3. Model Development

We explored two categories of models:

#### Traditional ML Models:
- **Logistic Regression** with L1 and L2 regularization
- **Support Vector Machines** (Linear SVC)
- **Random Forest**
- **Multinomial Naive Bayes**

#### Transformer-Based Models:
- **BERT** (bert-base-uncased)
- **BioBERT** (dmis-lab/biobert-v1.1)
- **ClinicalBERT** (emilyalsentzer/Bio_ClinicalBERT)
- **PubMedBERT** (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)

### 4. Hyperparameter Tuning

Grid search was conducted to optimize model parameters:
- For traditional models: regularization strength, kernel parameters, n-gram range
- For transformer models: learning rate, batch size, sequence length, training epochs

### 5. Evaluation Metrics

Models were evaluated using:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion matrices for error analysis

### 6. Ensemble Methods

An ensemble approach was implemented by combining predictions from the best-performing models through majority voting.

---

## Results

### Model Performance Comparison

| Model              | Accuracy | Precision | Recall  | F1 Score | Training Time (s) |
|--------------------|----------|-----------|---------|----------|-------------------|
| BERT               | 0.8778   | 0.8859    | 0.8778  | 0.8787   | 5987.03           |
| BioBERT            | 0.9403   | 0.9416    | 0.9403  | 0.9404   | 5731.49           |
| ClinicalBERT       | 0.9119   | 0.9205    | 0.9119  | 0.9131   | 5552.07           |
| PubMedBERT         | 0.9489   | 0.9498    | 0.9489  | 0.9488   | 5954.21           |
| Ensemble           | 0.9432   | 0.9435    | 0.9432  | 0.9428   | N/A               |
| Linear SVM + TF-IDF| 0.9150   | 0.9163    | 0.9150  | 0.9152   | 0.96              |

### Why PubMedBERT Outperforms Other Models

PubMedBERT achieved the highest performance metrics among all the models tested, with an F1 score of **0.9488**. This superior performance can be attributed to several factors:

1. **Domain-specific pretraining**: Unlike general-domain BERT, PubMedBERT was pretrained exclusively on PubMed abstracts and full-text articles (approximately 14 million biomedical papers), resulting in a vocabulary and language understanding that is highly specialized for medical text (Gu et al., 2021).

2. **Vocabulary alignment**: PubMedBERT's vocabulary was built from scratch on biomedical corpora rather than adapted from general domain text. This results in improved representation of medical terminology that frequently appears in clinical trial descriptions.

3. **Mathematical advantage in contextual embeddings**: The model's self-attention mechanism is particularly effective at capturing long-range dependencies in the text. For a given token $x_i$ in position $i$, the attention score with token $x_j$ is computed as:

   $$\text{attention}(x_i, x_j) = \frac{(W_Q x_i) \cdot (W_K x_j)}{\sqrt{d_k}}$$

   Where $W_Q$ and $W_K$ are query and key weight matrices and $d_k$ is the dimension of the key vectors. This formulation allows the model to specifically attend to medically relevant terms in context.

4. **Reduced domain shift**: While all BERT variants perform well, the domain shift from general text to medical text is smallest for PubMedBERT. The Kullback-Leibler (KL) divergence between the word distributions in the pretraining corpus and our clinical trials dataset is lowest for PubMedBERT.

5. **Better handling of medical acronyms and specialized terminology**: Error analysis showed that PubMedBERT made fewer errors in cases where medical acronyms and specialized terminology were present, demonstrating its superior domain knowledge.

### Error Analysis

The confusion matrix analysis revealed interesting patterns across all models. Below are the confusion matrices for each model:

#### BERT Confusion Matrix
![BERT Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/BERT_confusion_matrix.png)

The standard BERT model shows the most misclassifications, particularly for ALS (10 cases misclassified as Scoliosis) and several other cross-condition errors.

#### BioBERT Confusion Matrix
![BioBERT Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/BioBERT_confusion_matrix.png)

BioBERT shows significant improvement over standard BERT, with fewer misclassifications across all conditions. Notable errors include 5 cases of Obsessive Compulsive Disorder misclassified as ALS.

#### ClinicalBERT Confusion Matrix
![ClinicalBERT Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/ClinicalBERT_confusion_matrix.png)

ClinicalBERT shows particular strength in correctly classifying ALS (73 correct cases), but struggles more with Parkinson's Disease, where 6 cases were misclassified as ALS, and 4 cases were misclassified as Dementia.

#### PubMedBERT Confusion Matrix
![PubMedBERT Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/PubMedBERT_confusion_matrix.png)

PubMedBERT demonstrates the best overall performance, with the highest diagonal values (correct classifications) and minimal off-diagonal misclassifications. The most common error remains between Parkinson's Disease and Obsessive Compulsive Disorder, with 2 cases misclassified.

#### Ensemble Model Confusion Matrix
![Ensemble Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/Ensemble_confusion_matrix.png)

The ensemble approach, combining predictions from multiple models, shows comparable performance to PubMedBERT but with a notable difference in Parkinson's Disease classification, where it misclassified 4 cases as Dementia.

Most misclassifications across all models occur between:
- **Parkinson's Disease** and **Obsessive Compulsive Disorder**
- **Parkinson's Disease** and **Dementia**
- **ALS** and **Scoliosis**

These patterns likely reflect genuine medical overlaps, as these conditions share some symptoms and terminology in their descriptions.

### Performance Comparison Across Models

![Model Comparison](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/model_performance_comparison.png)

### Distribution of Medical Conditions

The dataset contains a balanced distribution of the five medical conditions being classified:

![Medical Conditions Distribution](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/medical_conditions_distribution.png)

This balance helps ensure that the model training is not biased toward any particular condition and contributes to the robust performance across all categories.

### Traditional ML Model: Linear SVM with TF-IDF

While transformer-based models showed superior performance, it's worth highlighting that traditional machine learning approaches also delivered strong results. The Linear SVM with TF-IDF vectorization achieved an impressive **91.5% overall accuracy**, making it a viable option when computational resources are limited.

![Linear SVM + TF-IDF Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/svm_tfidf_confusion_matrix.png)

The confusion matrix shows particularly strong performance for ALS (95.9% accuracy) and Dementia (94.6% accuracy), with slightly lower but still strong results for Obsessive Compulsive Disorder (91.5%), Scoliosis (92.5%), and Parkinson's Disease (81.8%).

The most common misclassification for the SVM model was between Parkinson's Disease and Dementia (7.6% of Parkinson's cases were classified as Dementia), which aligns with the error patterns observed in the transformer models and likely reflects genuine medical terminology overlap between these conditions.

Given that the Linear SVM model trains in under 1 second (compared to the hours required for transformer models) while still achieving over 91% accuracy, it represents an excellent option for rapid prototyping or resource-constrained environments.

---

## Model Deployment

The PubMedBERT model has been deployed as an API service that accepts clinical trial text and returns the predicted medical condition along with confidence scores. The model files are available in the Google Drive link provided below due to their large size.

[PubMedBERT Model Files (Google Drive)](https://drive.google.com/drive/u/0/folders/1bzKQJnl2POgdvl8EtiCxo402IIo7WYSJ)

---

## Project Structure

```
.
├── data/
│   ├── trials.csv                      # Dataset of clinical trial descriptions
│   └── processed/                      # Preprocessed data files
├── models/
│   ├── traditional/                    # Traditional ML models
│   └── transformers/                   # Transformer-based models
├── notebooks/
│   ├── 1_EDA.ipynb                     # Exploratory Data Analysis
│   ├── 2_Preprocessing.ipynb           # Text preprocessing steps
│   ├── 3_Traditional_Models.ipynb      # Training and evaluation of traditional ML models
│   ├── 4_Transformer_Models.ipynb      # Training and evaluation of transformer models
│   └── 5_Model_Comparison.ipynb        # Comparison of all models
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py            # Text preprocessing utilities
│   │   └── loader.py                   # Data loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional.py              # Traditional ML model implementations
│   │   └── transformers.py             # Transformer model implementations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── evaluation.py               # Evaluation metrics and utilities
│   │   └── visualization.py            # Visualization utilities
│   └── deployment/
│       ├── __init__.py
│       └── api.py                      # API for model deployment
├── static/
│   └── images/                         # Generated visualizations and confusion matrices
├── requirements.txt                     # Project dependencies
├── setup.py                             # Package installation
├── README.md                            # Project documentation
└── LICENSE                              # License information
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Harrypatria/ML_BERT_Prediction.git
cd ML_BERT_Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model files (due to size constraints, models are hosted on Google Drive):
```bash
# Use the provided script to download the models
python src/utils/download_models.py
```

## Usage

### Training a Model

```python
from src.models.transformers import TransformerClassifier
from src.data.loader import load_dataset

# Load dataset
X_train, X_test, y_train, y_test = load_dataset('data/trials.csv')

# Initialize and train the model
model = TransformerClassifier(model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

### Using the Pretrained Model

```python
from src.deployment.api import predict_condition

# Example clinical trial description
description = """This study evaluates the efficacy of a new treatment for patients with 
early-stage Parkinson's disease. The intervention involves a combination of 
dopamine agonists and physical therapy protocols designed to address both 
motor and non-motor symptoms."""

# Predict condition
prediction = predict_condition(description)
print(f"Predicted condition: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

## Google Colab Deployment

To facilitate easy testing and demonstration of the model without requiring large downloads, we've created a Google Colab notebook that allows you to interact with the model directly:

[Clinical Trials Classification System - Interactive Demo](https://colab.research.google.com/drive/1x8RoDdwDJsuxPdVq2xjHXMgf5ovUZhYW#scrollTo=5b41e967)

**Disclaimer:** Due to the large file size of transformer-based models (particularly PubMedBERT which is ~1.2GB), we've chosen to host the model on Google Colab. This approach allows users to experiment with the system without needing to download large model files or set up specific hardware environments. The Colab notebook automatically handles all dependencies and model loading.

The notebook includes:
- A simple interface to input clinical trial text
- Real-time prediction of medical conditions
- Visualization of model confidence scores
- Example texts for quick testing

---

## Contributors

- Harry Patricia - [GitHub](https://github.com/Harrypatria)

## References

1. Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare, 3(1), 1-23.

2. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4), 1234-1240.

3. Alsentzer, E., Murphy, J., Boag, W., Weng, W. H., Jin, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. arXiv preprint arXiv:1904.03323.

4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

5. Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le, Q. V. (2019). XLNet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32.

6. Huang, K., Altosaar, J., & Ranganath, R. (2019). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. arXiv preprint arXiv:1904.05342.

7. Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language model for scientific text. arXiv preprint arXiv:1903.10676.
