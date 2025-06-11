# MBTI Classification Task
## 1. Project Overview
This project classifies personality traits according to Myers-Briggs Type Indicator (MBTI) from text using PySpark for preprocessing and traditional machine learning and tensorflow for training a neural net based on BERT embeddings. It demonstrates scalable NLP pipelines and deep learning for classification. 

Be aware that although the MBTI is widely used it is also critized when it comes to validity. However I was interested, how well we can deduct from text to the personality of the author.

## 2. Dataset
- The Dataset contains user-generated posts labeled with MBTI personality types. They are the result of chats on a forum of the website personalitycafe, where users fill out the questionnaire for the MBTI and their result is publicly available. The Dataset is available on [Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type/data).
- Around 50 posts of about 8000 users were collected along with the persoanlity type of the author. We predict personality from single posts.

## 3. Approach
| Step        | Tool        | Description          |
|-------------|-------------|----------------------|
|1. Data Preprocessing & Class Machine Learning | PySpark, Pandas (for visualisation of small data parts) | Tokenization, filtering, feature extraction (TF-IDF, word count, etc.) |
|2. Training an MLP with BERT Embeddings | Tensorflow, Transformers | Train an MLP locally on a sample |
|3. Cloud Scaling | Google Cloud Vertex AI | Train a BERT model on the full training data on the Google Cloud |

## 4. Structure
- [notebook](Link to notebook) for demonstrating local training
- Files for Training on Google Cloud in [this subfolder](link to subfolder)

## 5. Results
Show a small table comparing different models (e.g., TF-IDF + Logistic Regression vs. LSTM vs. BERT).
Add some visualizations (e.g., confusion matrix, feature importance).
  
