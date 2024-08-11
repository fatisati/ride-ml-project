# Unfinished Ride Prediction

This project aims to predict unfinished rides for an online taxi service using various machine learning models.

## Project Structure

- `data/`: Contains the processed data files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `src/`: Source code for data preprocessing, model training, and evaluation.

## Installation

1. Create a virtual environment and activate it.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Locally

1. **Preprocess the data**:
    ```bash
    python src/data_preprocessing.py
    ```

2. **Train and evaluate models**:
    ```bash
    python src/train.py
    python src/evaluate.py
    ```

3. **Make predictions and save results**:
    ```bash
    python src/predict_and_save.py
    ```

### Running in Google Colab

You can also run the entire project in a Google Colab notebook, which includes preprocessing, training, evaluation, and prediction:

- **Google Colab Link**: [Run in Colab](https://colab.research.google.com/drive/1dQP5MAypkkgaPMTr8jOk9cuT1XkaWl-7#scrollTo=pU5wywc9wlHP)

The Colab notebook includes classification results for all three models (Logistic Regression, Random Forest, and Deep Neural Network), providing an easy and interactive way to explore the project.
