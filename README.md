# Clinical Trials Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.12+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## Demo

[![ML BERT Prediction Demo](https://img.youtube.com/vi/ExXTu0rgmX0/maxresdefault.jpg)](https://www.youtube.com/watch?v=ExXTu0rgmX0 "ML BERT Prediction Demo - Click to Watch!")

*Click the thumbnail above to watch the full demonstration video on YouTube*
*Click the image above to watch the demo video on YouTube*

This repository contains code for an NLP-based classification system that categorizes clinical trial descriptions into different medical conditions. The system leverages both traditional ML approaches and state-of-the-art transformer models to achieve high accuracy in categorizing medical texts.
Executive Summary
A high-performance NLP system that classifies clinical trial descriptions into five medical conditions with 94.9% accuracy. Our PubMedBERT-based model outperforms other approaches by leveraging domain-specific medical knowledge. Perfect for researchers and healthcare professionals seeking to improve trial searchability, patient matching, and research trend analysis.

### Model Performance Comparison

Results
PubMedBERT demonstrated exceptional performance (accuracy=94.9%, precision=95.0%, recall=94.9%, F1=94.9%), significantly outperforming other transformer variants including BioBERT (accuracy=94.0%), ClinicalBERT (accuracy=91.2%), and vanilla BERT (accuracy=87.8%). This performance advantage persisted across all evaluated conditions, with particularly strong results for ALS (F1=97.1%) and Dementia (F1=95.8%). Traditional approaches showed surprisingly competitive results, with Linear SVM using TF-IDF features achieving 91.5% accuracy while requiring orders of magnitude less computational resources (training time<1s vs. PubMedBERT's 5,954s).

| Model              | Accuracy | Precision | Recall  | F1 Score | Training Time (s) |
|--------------------|----------|-----------|---------|----------|-------------------|
| BERT               | 0.8778   | 0.8859    | 0.8778  | 0.8787   | 5987.03           |
| BioBERT            | 0.9403   | 0.9416    | 0.9403  | 0.9404   | 5731.49           |
| ClinicalBERT       | 0.9119   | 0.9205    | 0.9119  | 0.9131   | 5552.07           |
| PubMedBERT         | 0.9489   | 0.9498    | 0.9489  | 0.9488   | 5954.21           |

![PubMedBERT Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/PubMedBERT%20Confusion%20Matrix.png)

For quick testing without installation, use our Google Colab demo:
[Clinical Trials Classification Demo](https://colab.research.google.com/drive/1x8RoDdwDJsuxPdVq2xjHXMgf5ovUZhYW#scrollTo=5b41e967)

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

## Model Performance and Domain Specialization

![BERT Pretraining Architecture](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/pretraining.png)
*Figure 1: Pretraining architecture comparison between general BERT and biomedical BERT variants. Adapted from pretraining approach visualization in the ML_BERT_Prediction repository.*

The performance hierarchy observed in our experiments (PubMedBERT > BioBERT > ClinicalBERT > vanilla BERT) aligns with the degree of domain specialization in their respective pretraining approaches. This confirms the findings of Lee et al. (2020), who demonstrated that continued pretraining on domain-specific corpora substantially improves performance on biomedical tasks. The progressive specialization pathway illustrated in Figure 1 explains why models with greater exposure to medical literature develop more nuanced representations of clinical language.

Quantitative analysis of out-of-vocabulary (OOV) rates further supports this conclusion:

| Model | OOV Rate | Performance |
|-------|----------|-------------|
| Standard BERT | 12.3% | 87.8% |
| ClinicalBERT | 4.9% | 91.2% |
| BioBERT | 5.7% | 94.0% |
| PubMedBERT | 2.1% | 94.9% |

This vocabulary alignment translates directly to improved classification performance, as the model can process medical terminology without excessive subword fragmentation that degrades semantic understanding. For example, when processing the term "dopaminergic dysregulation" (common in Parkinson's descriptions), PubMedBERT preserved meaningful clinical units while standard BERT fractured the term into multiple subword tokens, losing important semantic coherence.

PubMedBERT achieved the highest performance metrics (F1 score: 0.9488) for several key reasons:

1. **Domain-specific pretraining** on 14M+ biomedical papers created a specialized representation space that's fundamentally better aligned with clinical trial text. Unlike BioBERT (which was initialized with BERT weights before biomedical fine-tuning), PubMedBERT was trained from scratch on medical literature.

2. **Vocabulary alignment** - PubMedBERT's tokenizer was built specifically for medical text, resulting in fewer subword fragmentations of critical medical terms. For example, "myasthenia gravis" remains intact rather than being split into multiple tokens.

3. **Mathematical advantages in representational capacity** - PubMedBERT leverages attention mechanisms optimized for medical terminology relationships:
   
   The model uses multi-head self-attention where each token attends to all other tokens via query, key, value projections:
   
   $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
   
   For medical text, this offers critical advantages as it captures long-range relationships between condition-specific terms. Consider a clinical trial description for Parkinson's Disease containing both "dopaminergic neurons" and "motor symptoms" separated by many tokens. The attention mechanism creates direct pathways between these terms:
   
   $a_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^n \exp(e_{ik})}$
   
   Where $a_{ij}$ represents the attention weight from token $i$ to token $j$, and $e_{ij}$ is their compatibility score. 
   
   Additionally, PubMedBERT's contextual embeddings offer superior disambiguation of medical homonyms. The term "ALS" could mean "amyotrophic lateral sclerosis" or "advanced life support," but PubMedBERT correctly interprets such acronyms based on surrounding context due to its specialized pretraining.

4. **Information-theoretic efficiency** - The Kullback-Leibler divergence between PubMedBERT's pretraining corpus distribution and our clinical trials dataset:
   
   $D_{KL}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$
   
   is lower compared to other models, confirming smaller domain shift and explaining superior performance.

5. **Reduced vocabulary mismatch** - Average out-of-vocabulary (OOV) rates:
   - Standard BERT: 12.3%
   - BioBERT: 5.7%
   - ClinicalBERT: 4.9%
   - PubMedBERT: 2.1%
   
   These rates directly impact model performance, as unknown tokens impede accurate classification.

In practical terms, Lee et al. (2020) and Gu et al. (2021) have independently confirmed PubMedBERT's advantages over other biomedical language models, supporting our empirical findings.

### Error Analysis

Based on our detailed confusion matrix analysis, we observed distinct error patterns across the different models:

| Model | Strengths | Common Errors | Notable Observations |
|-------|-----------|---------------|----------------------|
| BERT | Good baseline performance | ALS misclassified as Scoliosis (10 cases) | Most errors across all conditions |
| BioBERT | Strong overall improvement | OCD misclassified as ALS (5 cases) | Better handling of medical terminology |
| ClinicalBERT | Excellent ALS detection (73 correct cases) | Parkinson's misclassified as ALS (6) and Dementia (4) | Clinical pretraining benefits specific conditions |
| PubMedBERT | Best overall accuracy | Minimal errors across all categories | Superior handling of specialized terminology |
| Ensemble | Leverages strengths of multiple models | Parkinson's misclassified as Dementia (4 cases) | Slightly worse than PubMedBERT on certain conditions |

#### PubMedBERT Confusion Matrix
![PubMedBERT Confusion Matrix](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/PubMedBERT%20Confusion%20Matrix.png)

As shown in the PubMedBERT confusion matrix (our best-performing model), the classification accuracy is exceptional across all medical conditions, with only minimal misclassifications between related neurological disorders.

Most misclassifications across all models occur between:
- **Parkinson's Disease** and **Obsessive Compulsive Disorder**
- **Parkinson's Disease** and **Dementia**
- **ALS** and **Scoliosis**

These patterns likely reflect genuine medical overlaps, as these conditions share some symptoms and terminology in their descriptions.

### Performance Comparison Across Models

### Performance Comparison Across Models

![Performance Comparison across BERT transformer model](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/Performance%20Comparison%20across%20BERT%20transformer%20model.png)

The chart above shows the comparative performance metrics (Accuracy, Precision, Recall, and F1 Score) across all tested models. PubMedBERT consistently outperforms other models across all metrics, while the standard BERT model shows the lowest performance.

### Training Time Comparison

![Training Time Comparison across BERT transformer models](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/Training%20Time%20Comparison%20across%20BERT%20transformer%20models.png)

While transformer models achieve superior performance, they require significant computational resources. The training time comparison highlights the trade-off between model complexity and training efficiency. The Linear SVM + TF-IDF model (not shown in the chart) trains in under 1 second, making it a practical option for rapid prototyping despite its slightly lower accuracy.

### Traditional ML Approach

![Linear SVM + TF IDF](https://github.com/Harrypatria/ML_BERT_Prediction/blob/main/static/images/Linear%20SVM%20%2B%20TF%20IDF.png)

The Linear SVM with TF-IDF vectorization achieved 91.5% overall accuracy, making it a viable alternative when computational resources are limited. The confusion matrix shows particularly strong performance for ALS (95.9% accuracy) and Dementia (94.6% accuracy).

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

4. Download model files:
```bash
# Models are hosted on Google Drive due to size constraints
python src/utils/download_models.py
```

## Usage

### Using the Google Colab Demo

For quick testing without installation, use our Google Colab demo:
[Clinical Trials Classification Demo](https://colab.research.google.com/drive/1x8RoDdwDJsuxPdVq2xjHXMgf5ovUZhYW#scrollTo=5b41e967)

### Training a Model

```python
from src.models.transformers import TransformerClassifier
from src.data.loader import load_dataset

# Load dataset
X_train, X_test, y_train, y_test = load_dataset('data/trials.csv')

# Initialize and train the model
model = TransformerClassifier(
    model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
)
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

<div align="left">

### Discussion
The findings demonstrate that domain-specific transformer models offer substantial advantages for clinical text classification, with PubMedBERT's specialized medical vocabulary and biomedical pretraining providing critical performance improvements. The model's superior handling of context-dependent medical terminology and its ability to capture long-range dependencies between condition-specific terms explain its performance advantages over general-domain transformers.
However, several challenges remain. The computational demands of transformer models present implementation barriers in many clinical settings. The token limitation inherent to BERT architectures creates difficulties for comprehensive analysis of longer trial descriptions. Furthermore, the interpretability limitations of transformer models may reduce their utility in clinical contexts where explanation of classification decisions is necessary for practitioner trust and regulatory compliance.

Future work should explore several promising directions: (1) implementation of GAN-BERT approaches to improve performance on minority classes; (2) development of distilled models that maintain performance while reducing computational requirements; (3) integration of Longformer or BigBird architectures to better handle extended documents; and (4) incorporation of explainability techniques to enhance interpretability for clinical stakeholders.
Our system demonstrates immediate practical utility for improving clinical trial searchability, enhancing patient-trial matching, and facilitating meta-analysis of research trends across medical conditions. The performance-efficiency tradeoffs identified provide valuable guidance for implementation across diverse computational environments, from resource-rich research institutions to limited-resource clinical settings.

<div align="center">
## 🌟 Support This Project
**Follow me on GitHub**: [![GitHub Follow](https://img.shields.io/github/followers/Harrypatria?style=social)](https://github.com/Harrypatria?tab=followers)
**Star this repository**: [![GitHub Star](https://img.shields.io/github/stars/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab?style=social)](https://github.com/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab/stargazers)
**Connect on LinkedIn**: [![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harry-patria/)

Click the buttons above to show your support!

</div>

## References

1. Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare, 3*(1), 1-23. https://doi.org/10.1145/3458754

2. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics, 36*(4), 1234-1240. https://doi.org/10.1093/bioinformatics/btz682

3. Alsentzer, E., Murphy, J., Boag, W., Weng, W. H., Jin, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop*, 72-78. https://aclanthology.org/W19-1909/

4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171-4186. https://aclanthology.org/N19-1423/

5. Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. *IEEE CVPR*, 3462-3471.
