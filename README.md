# 📊 Project Title: *Advanced House Prices Prediction*

## Overview

This project predicts house prices using the Ames Housing dataset. It demonstrates a full machine learning workflow: data cleaning, feature engineering, model training, and deployment via a Streamlit web app. The goal is to provide accurate price predictions and a reproducible pipeline for advanced regression tasks.

---


## 📁 Project Structure

```
├── data/                # Datasets and description
│   ├── train.csv        # Training data
│   ├── test.csv         # Test data
│   ├── new_train.csv    # Cleaned/engineered train data
│   ├── new_test.csv     # Cleaned/engineered test data
│   ├── sample_submission.csv
│   └── data_description.txt
│
├── models/              # Trained model(s)
│   └── linear_regression_model.pkl
│
├── configs/             # App configuration, mostly for the streamlit app
│   └── config.py
│
├── views/               # Streamlit app logic
│   ├── house_price.py   # Main house price prediction UI
│   └── custom_app.py    # Custom regression UI
│
├── Data_Cleaning_and_Feature_Engineering_Final_Version.ipynb # Notebook for data cleaning & feature engineering
├── main.py              # Streamlit entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization
├── start.sh             # Shell script to launch app with docker
├── LICENSE.txt
└── README.md            # This file
```

---

## 🔧 Setup and Installation Instructions


### 1. Clone the repository

```bash
git clone <repo-url>
cd Kaggle_Advanced_House_Prices
```


### Option.1 Create and activate a virtual environment

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bat
python -m venv .venv
.\.venv\Scripts\activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### Option.2 Run in Docker

```bash
docker build -t streamlit_app .
docker run --rm -p 8501:8501 streamlit_app

```
Or you can use the bash script to run these commands for you using the following command:

```bash
bash start.sh
```

---


## 🛠️ Step-by-Step Guide

### Step 1: Data Cleaning & Feature Engineering
- Open and run `Data_Cleaning_and_Feature_Engineering_Final_Version.ipynb` to preprocess and engineer features from the raw data.
- Follow up to train a Linear Regression Model on this difficult case.

### Step 2: Train Model (if needed)
- Use scripts or notebook to train and save models in `models/` (default model provided).

### Step 3: Launch the Streamlit App
- Run the following command:
	```bash
	streamlit run main.py
	```
- Use the sidebar to select between house price prediction and custom regression modules.
- Train your own model on your own data using the custom regression module.


---


## 📊 Results

- **Predictions:** View predicted house prices in the Streamlit app UI.
- **Model:** The trained linear regression model is stored in `models/linear_regression_model.pkl`.
- **Data:** Cleaned datasets are in `data/new_train.csv` and `data/new_test.csv`.
- **Notebook:** All preprocessing and feature engineering steps are documented in the notebook.
- **Evaluation Metrics:**  
    - *Mean Absolute Error (MAE):* Measures the average absolute difference between predicted and actual prices. Lower values indicate better accuracy.
    - *Mean Squared Error (MSE):* Measures the average squared difference between predicted and actual prices. It penalizes larger errors more than MAE.  
        **Notebook Result:** 711,102,117.35

For more details, see comments in code and notebook sections.
