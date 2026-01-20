# Classifying MBTI Personality Types from Text

*by Moritz Grimm*

Can we infer a person’s MBTI type from how they write online? 
This project explores exactly that, using a dataset of 300k+ user-generated posts and comparing both classic machine learning and transformer-based NLP models.

---

## Highlights

- Preprocessing of noisy social media text
- Word & character features + word count-based models
- BERT embeddings with a modular PyTorch training pipeline
- Ablation study & class imbalance strategies
- Hyperparameter optimization with Optuna
- Full-scale training on Google Cloud Vertex AI
- Critical sociological and methodological reflections on the dataset and the MBTI concept

---

## Executive Summary

This project explores the prediction of MBTI personality types based on user-generated text.  
Both classical models and transformer-based methods are used, with a focus on reproducibility, automation, and fair evaluation under class imbalance.

A fully documented pipeline supports:
- Feature importance analysis  
- End-to-end reproducibility  
- Modular pipeline supporting both local and cloud-based training

Although it is difficult to make reliable predictions of the MBTI type, the best-performing model achieved a test accuracy of 20.95%, which is more than three times better than random guessing (6.25%) in a 16-class setting.

---

## Resources

- [Rendered HTML version of the Notebook](https://ryoshuu.github.io/Portfolio/mbti/mbti_nb.html) (The executable notebook is maintained separately and available on request)
- [Cloud Training Scripts on GitHub](https://github.com/Ryoshuu/Portfolio/tree/main/mbti-classification/GCP)


---

## Motivation

This project is part of my personal portfolio and a larger interest in applying NLP to human-generated text data.  
I am especially interested in transparent, reproducible machine learning — and in understanding how (and if) personality manifests in text.

---

## Technologies

- Python, PySpark, PyTorch, Pandas
- Matplotlib (Pyplot) and Seaborn for visualization
- Hugging Face Transformers
- Optuna for hyperparameter tuning
- Google Cloud Vertex AI for scalable training

  







