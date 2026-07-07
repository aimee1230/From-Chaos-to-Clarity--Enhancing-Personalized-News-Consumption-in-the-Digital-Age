# News Classification and Summarization using Machine Learning and Deep Learning

This repository contains the implementation of an intelligent news analysis system that combines **news classification** and **text summarization** to help users quickly understand large volumes of online news. The project evaluates multiple machine learning algorithms and word embedding techniques for news classification and extends the work by integrating a **BART transformer model** for abstractive summarization through a simple web application.

---

## Overview

With the exponential growth of digital news platforms, readers are often overwhelmed by the volume of information available online. This project addresses this challenge by automatically classifying news articles into predefined categories and generating concise summaries, enabling faster and more personalized information consumption.

The project was initially developed as a comparative study of classical machine learning models for news classification and extractive summarization. It was later extended with a transformer-based summarization model and deployed as a web application for an end-to-end user experience.

---

## Features

- Automatic news article classification
- Extractive summarization using TextRank and LexRank
- Abstractive summarization using the BART transformer model
- Comparison of multiple machine learning algorithms
- Evaluation of various word embedding techniques
- Interactive web application for real-time news analysis

---

## Machine Learning Models

- Support Vector Machine (SVM)
- Logistic Regression (LR)
- Multinomial Naive Bayes (MNB)
- Random Forest (RF)
- Decision Tree Classifier (DTC)

---

## Word Embedding Techniques

- TF-IDF
- Word2Vec
- GloVe
- FastText
- Continuous Bag of Words (CBOW)
- Skip-Gram

---

## Summarization Models

### Extractive Summarization

- TextRank
- LexRank

### Abstractive Summarization

- BART

---

## Datasets

The project was evaluated using three publicly available datasets:

- CNN News Articles (2011–2022)
- News Article Category Dataset
- BBC News Summary Dataset

---

## Results

- Best Performing Models:
  - Support Vector Machine (SVM)
  - Logistic Regression (LR)

- Best Feature Representation:
  - TF-IDF

- Classification Performance:
  - Precision: **88%**
  - Recall: **88%**

The extended system integrates a fine-tuned BART model to generate fluent abstractive summaries and provides an intuitive web interface where users can submit a news article, automatically classify it, and receive a concise summary.

---

## Technologies Used

- Python
- Scikit-learn
- Hugging Face Transformers
- BART
- Pandas
- NumPy
- NLTK
- Gensim
- Flask
- HTML/CSS
- JavaScript

---

## Future Improvements

- Fine-tuning transformer models on domain-specific news datasets
- Multi-language news classification and summarization
- Personalized news recommendations
- Deployment using Docker and cloud platforms
