# Forecasting Grocery Sales Data

*by Moritz Grimm*

Predicting the future is one of humanity’s oldest ambitions. In this project, I forecast sales for the Ecuadorian grocery chain Favorita across all 54 stores and 33 product families for the next 14 days, using more than 3 million historical daily observations to train classical tree-based models as well as a transformer-based neural network.

---

## Highlights

- Feature engineering from date, holiday, and event data  
- Clustering 1,784 time series by their shape  
- A custom fully vectorized recursive LightGBM model using lagged features  
- Training a Temporal Fusion Transformer with PyTorch Forecasting  
- Leakage-aware, modular scikit-learn pipelines and time-aware cross-validation  
- Hyperparameter optimization with Optuna  
- Sociological and personal reflections on the dataset and the fascination of forecasting itself  

---

## Executive Summary

This project explores grocery sales forecasting using both classical models and transformer-based methods, with a strong focus on reproducibility, automation, and fair evaluation.

A fully documented pipeline supports:
- Feature engineering  
- End-to-end reproducibility  
- A modular design covering both classical and deep learning approaches  
- Systematic model selection  

Although the dataset is noisy and affected by unpredictable spikes and regime shifts, the most reliable model reduced the naïve baseline error by **31.1% on the test set** and by **19% on average across all five cross-validation folds** within the training window, making it a strong candidate for production use.

---

## Resources

- [Rendered HTML version of the notebook](https://ryoshuu.github.io/Portfolio/favorita-sales-forecasts/favorita_nb.html)  
  *(The executable notebook is maintained separately and available on request.)*

---

## Motivation

This project is part of my personal portfolio and reflects a broader interest in time series forecasting.  
I am particularly interested in transparent, reproducible machine learning — and in understanding how (and if) reality manifests itself in time series data.

---

## Technology

- Python, Pandas, scikit-learn, PyTorch  
- Matplotlib (pyplot) and Seaborn for visualization  
- Optuna for hyperparameter tuning  
