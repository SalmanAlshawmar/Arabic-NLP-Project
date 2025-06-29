# Arabic NLP Project – KALIMAT Dataset

A comprehensive Natural Language Processing project exploring Arabic text classification and summarization using both traditional machine learning and modern transformer-based approaches.

## Project Overview

This project implements and compares two core NLP tasks on Arabic text data:
- **Text Classification**: Topic classification using the KALIMAT corpus
- **Text Summarization**: Both extractive and generative summarization approaches

The goal is to evaluate traditional machine learning methods against modern deep learning/transformer-based models, highlighting how each approach handles Arabic-specific linguistic challenges.

## Datasets

1. **KALIMAT Corpus** (Text Classification)
   - Arabic texts categorized into 6 topics: culture, economy, local news, international news, religion, and sports
   - [Download the KALIMAT dataset](https://sourceforge.net/projects/kalimat/files/kalimat/document-collection/)

2. **Arabic Summarization Corpus** (Text Summarization)
   - Arabic texts paired with corresponding summaries

## Technologies Used

- **Traditional ML**: SVM, TF-IDF vectorization
- **Deep Learning**: LSTM, Seq2seq model with attention
- **Transformers**: BERT-based models for Arabic
- **Libraries**:
  - scikit-learn, NLTK
  - PyTorch, Transformers (Hugging Face)
  - pandas, numpy, matplotlib, seaborn

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arabic-nlp-project.git
cd arabic-nlp-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook arabic_nlp_project.ipynb
```

## Project Structure

```
arabic-nlp-project/
├── README.md
├── requirements.txt
├── arabic_nlp_project.ipynb
├── data/                    # (Not included - download separately)
│   ├── kalimat/
│   └── summarization/
└── results/                 # Generated outputs and models
    ├── models/
    ├── plots/
    └── reports/
```

## Project is divided into two phases:

### Phase 1: Traditional Approaches

**Text Classification:**
- Text preprocessing and normalization for Arabic
- TF-IDF feature extraction for document representation
- SVM classification using TF-IDF features

**Text Summarization:**
- TF-IDF based sentence scoring and ranking
- Extractive summarization using TF-IDF weights
- Sentence selection based on TF-IDF importance scores

### Phase 2: Modern Deep Learning Approaches

**Text Classification:**
- LSTM-based classification
- Transformer models (BERT for Arabic)

**Text Summarization:**
- Sequence-to-sequence models with attention
- Pre-trained Arabic language models for summarization

### Comparative Analysis
- Performance metrics comparison
- Computational efficiency analysis
- Handling of Arabic-specific challenges

## Results

### Text Classification Performance

| Model | Approach | F1-Score |
|-------|----------|----------|
| SVM + TF-IDF | Traditional | 93% |
| LSTM | Deep Learning | 86.4% |
| BERT | Transformer | 89.2% |



### Key Findings

- **Classification**: Traditional SVM + TF-IDF achieved the highest performance (93% F1-score), demonstrating the effectiveness of well-tuned traditional methods for Arabic text classification
- **Deep Learning vs Traditional**: LSTM (86.4%) and BERT (89.2%) performed lower than the traditional approach, possibly due to limited training data or hyperparameter tuning
- **Summarization**: Arabic-specific transformers achieve the best performance across all metrics
- **Efficiency**: Traditional methods are fastest and most effective for this classification task
- **Arabic Handling**: Proper preprocessing and TF-IDF feature extraction proved highly effective for Arabic morphology

## Usage

Open `arabic_nlp_project.ipynb` in Jupyter Notebook or JupyterLab and run the cells sequentially. The notebook is organized into clear sections:

1. Data Exploration and Preprocessing
2. Traditional ML Approaches
3. Deep Learning and Transformer Approaches
4. Comparative Analysis