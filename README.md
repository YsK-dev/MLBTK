# MLBTK - Machine Learning and NLP Toolkit

## Overview

This repository contains comprehensive implementations of machine learning algorithms and natural language processing techniques. It includes practical examples, educational content, and advanced applications covering both traditional ML methods and modern deep learning approaches.

## Repository Structure

### 📊 **ML-Sklearn** - Machine Learning Algorithms
Implementation of core machine learning algorithms using scikit-learn and related libraries:

- **Supervised Learning**: Classification and regression techniques
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Model Evaluation**: Hyperparameter tuning and cross-validation
- **Advanced Projects**: Including comprehensive Titanic survival prediction with network analysis

### 🔤 **NLP** - Natural Language Processing
Comprehensive NLP implementations from basic to advanced:

- **Text Preprocessing**: Tokenization, cleaning, and feature extraction
- **Traditional NLP**: N-grams, TF-IDF, and bag-of-words
- **Deep Learning**: RNNs, LSTMs, and Transformer models
- **Applications**: Sentiment analysis, text classification, machine translation, and recommendation systems

## 📊 Machine Learning - ML-Sklearn

### Classification Algorithms
- **[1_KNN.py](Ml-Sklearn/1_KNN.py)** - K-Nearest Neighbors implementation with breast cancer classification
- **[2_decisionTrees.py](Ml-Sklearn/2_decisionTrees.py)** - Decision Tree classifier and regressor with visualization
- **[3_randomForest.py](Ml-Sklearn/3_randomForest.py)** - Random Forest for classification and regression
- **[4_logisticRegression.py](Ml-Sklearn/4_logisticRegression.py)** - Logistic regression with heart disease prediction
- **[5_SVM.py](Ml-Sklearn/5_SVM.py)** - Support Vector Machine with digit classification
- **[6_NaiveBayes.py](Ml-Sklearn/6_NaiveBayes.py)** - Gaussian Naive Bayes classifier
- **[7_ClassificationModelComparision.py](Ml-Sklearn/7_ClassificationModelComparision.py)** - Comprehensive model comparison

### Regression Algorithms
- **[9_linearRegression.py](Ml-Sklearn/9_linearRegression.py)** - Linear and polynomial regression implementations

### Unsupervised Learning
- **[10_KMeansClustering.py](Ml-Sklearn/10_KMeansClustering.py)** - K-means, hierarchical, DBSCAN, and other clustering algorithms
- **[12_PCA.py](Ml-Sklearn/12_PCA.py)** - Principal Component Analysis and Linear Discriminant Analysis with t-SNE

### Advanced Topics
- **[11_QLearning.py](Ml-Sklearn/11_QLearning.py)** - Q-Learning reinforcement learning implementation
- **[13_hyperparametertunning.py](Ml-Sklearn/13_hyperparametertunning.py)** - Grid search vs random search optimization
- **[14_L1andL2Regularization.py](Ml-Sklearn/14_L1andL2Regularization.py)** - Ridge, Lasso, and Elastic Net regularization

### Featured Project
- **[8_titanicEda.py](Ml-Sklearn/8_titanicEda.py)** - Advanced Titanic survival prediction with:
  - Multi-layered social network analysis
  - Quantum-inspired feature engineering
  - Advanced probabilistic clustering
  - Deep neural network embeddings
  - Meta-learning ensemble architecture

## 🔤 Natural Language Processing - NLP

### Text Preprocessing & Feature Extraction
- **[cleaning.py](NLP/cleaning.py)** - Text cleaning and preprocessing utilities
- **[test_tokenization.py](NLP/test_tokenization.py)** - Tokenization techniques
- **[stemming_lemmatization.py](NLP/stemming_lemmatization.py)** - Text normalization methods
- **[stop_words.py](NLP/stop_words.py)** - Stop words handling
- **[bag_of_words.py](NLP/bag_of_words.py)** - Bag of Words implementation with IMDB dataset
- **[N_Gram.py](NLP/N_Gram.py)** - N-gram analysis (unigram, bigram, trigram)
- **[tf_ıdf.py](NLP/tf_ıdf.py)** - TF-IDF vectorization

### Word Representations
- **[word_embedings.py](NLP/word_embedings.py)** - Word2Vec and FastText implementations with clustering
- **[word_meaning_unneccarity.py](NLP/word_meaning_unneccarity.py)** - Word sense disambiguation using Lesk algorithm

### Deep Learning for NLP
- **[rnn.py](NLP/rnn.py)** - RNN implementation for sentiment analysis
- **[lstm.py](NLP/lstm.py)** - LSTM for text generation and analysis
- **[nlp_trandformers.py](NLP/nlp_trandformers.py)** - BERT and transformer implementations
- **[transfromers.py](NLP/transfromers.py)** - GPT-2 and Llama text generation

### Applications
- **[sentiment_analysis.py](NLP/sentiment_analysis.py)** - VADER sentiment analysis on IMDB reviews
- **[text_classification.py](NLP/text_classification.py)** - Spam classification using various algorithms
- **[text_summ.py](NLP/text_summ.py)** - Text summarization using transformers
- **[machine_translation.py](NLP/machine_translation.py)** - Neural machine translation with MarianMT
- **[name_entity_recognition.py](NLP/name_entity_recognition.py)** - NER and POS tagging with spaCy
- **[qa_bert.py](NLP/qa_bert.py)** - Question answering with BERT and GPT
- **[info_retriev.py](NLP/info_retriev.py)** - Information retrieval using BERT embeddings
- **[recommedation_system.py](NLP/recommedation_system.py)** - Collaborative filtering with neural networks and surprise library
- **[chatbot.py](NLP/chatbot.py)** - Chatbot implementation
- **[max_entropy.py](NLP/max_entropy.py)** - Maximum entropy classifier
- **[hidden_markov.py](NLP/hidden_markov.py)** - Hidden Markov Models

## Key Features & Highlights

### 🎯 Advanced Machine Learning Techniques
- **Ensemble Methods**: Voting classifiers and stacking architectures
- **Hyperparameter Optimization**: Grid search and random search implementations
- **Cross-validation**: K-fold and Leave-One-Out validation strategies
- **Regularization**: L1, L2, and Elastic Net for overfitting prevention
- **Dimensionality Reduction**: PCA, LDA, and t-SNE visualizations

### 🚀 Cutting-Edge NLP Applications
- **Transformer Models**: BERT, GPT-2, and MarianMT implementations
- **Deep Learning**: RNN and LSTM architectures for text analysis
- **Word Embeddings**: Word2Vec and FastText with clustering visualization
- **Information Retrieval**: Semantic search using BERT embeddings
- **Multi-modal Analysis**: Text classification, sentiment analysis, and generation

### 📈 Featured Projects

#### Titanic Survival Prediction (Advanced)
- Multi-layered social network analysis
- Quantum-inspired feature engineering
- Advanced clustering and anomaly detection
- Meta-learning ensemble with uncertainty quantification

#### Recommendation System
- Neural collaborative filtering
- Matrix factorization with embeddings
- User-based and item-based filtering
- MovieLens dataset implementation

#### Comprehensive Text Analysis Pipeline
- End-to-end preprocessing and feature extraction
- Multiple classification algorithms comparison
- Deep learning sentiment analysis
- Information retrieval and question answering

This repository serves as:
- **Learning Resource**: Step-by-step implementations with detailed comments
- **Reference Material**: Best practices for ML and NLP workflows
- **Project Templates**: Reusable code for common tasks
- **Advanced Techniques**: Cutting-edge methods for research and development

## Datasets Used

- **IMDB Movie Reviews**: Sentiment analysis and text classification
- **Spam SMS Dataset**: Binary classification example
- **Titanic Dataset**: Survival prediction with advanced feature engineering
- **MovieLens**: Recommendation system implementation
- **Breast Cancer Wisconsin**: Medical diagnosis classification
- **California Housing**: Regression analysis
- **Iris Dataset**: Multi-class classification
- **Digits Dataset**: Image classification with SVM

## Contributing

Feel free to contribute by:
- Adding new algorithms or techniques
- Improving existing implementations
- Adding more datasets and examples
- Enhancing documentation and comments
- Reporting bugs or suggesting improvements

## Future Enhancements

### Planned Additions
1. **Computer Vision**: CNN implementations and image processing
2. **Time Series Analysis**: ARIMA, LSTM for temporal data
3. **Reinforcement Learning**: Extended Q-learning and policy gradient methods
4. **Graph Neural Networks**: Advanced network analysis techniques
5. **MLOps**: Model deployment and monitoring examples
6. **Explainable AI**: SHAP values and model interpretability tools

---

*This repository demonstrates practical implementations of machine learning and NLP techniques, from fundamental algorithms to state-of-the-art deep learning models, providing a comprehensive learning resource for data science enthusiasts and practitioners.*