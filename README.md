# 🏠 Advanced House Prices Prediction

A complete machine learning solution for predicting house prices using the Ames Housing dataset. This project implements an end-to-end pipeline from data preprocessing to web deployment, achieving competitive performance on the Kaggle leaderboard.

## 🚀 Features

- **Data Processing Pipeline**: Automated cleaning and feature engineering for optimal model performance. Please read `data/data_description.txt` for detailed feature explanations and comprehensive understanding of the dataset.
- **Multiple ML Models**: Linear Regression and XGBoost implementations with performance comparison
- **Interactive Web Application**: Streamlit-based interface for real-time predictions and custom model training
- **Production Ready**: Docker containerization and comprehensive evaluation metrics

## 📂 Folder Structure

```
├── 📂 data/                # Datasets and description
│   ├── 📄 train.csv        # Training data
│   ├── 📄 test.csv         # Test data
│   ├── 📄 new_train.csv    # Processed training data
│   ├── 📄 new_test.csv     # Processed test data
│   ├── 📄 data_description.txt
├── 📂 models/              # Trained models and schemas
│   ├── 📦 linear_regression_model.pkl
│   └── 📄 schemas.py       # Pydantic data validation
│
├── 📂 configs/             # Application configuration
│   └── ⚙️ config.py
│
├── 📂 views/               # Streamlit application components
│   ├── 🏡 house_price.py   # Main prediction interface
│   ├── 🛠️ custom_linear_app.py    # Custom model training
│   └── 🚀 custom_xgboost.py       # XGBoost implementation
│
├── 📓 data_cleaning_and_feature_engineering.ipynb
├── 🐳 Dockerfile           # Container configuration
├── environment.yml         # Conda environment
├── Instructions.md         # Student instructions
├── LICENSE                 # License information
├── 🚀 main.py              # Application entry point
├── Party_Time.ipynb        # Google Colab notebook
├── 📦 requirements.txt     # Dependencies
├── server-instructions.md  # Server setup guide
└── 🖥️ start.sh             # Launch script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- Kaggle account (for dataset access)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Kaggle_Advanced_House_Prices
   ```

---

## ⚙️ Environment Setup
### 🅰️ Option 1: Create and activate a virtual environment

#### 🐧 Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
#### 🪟 Windows

```bat
python -m venv .venv
.\.venv\Scripts\activate
```

Or alternatively using Conda for all OS:

```bash
conda env create -f environment.yml
conda activate kaggle-house-prices
```

#### Add conda to your kernel to use it in Jupyter Notebook.

   ```bash
   conda install ipykernel
   python -m ipykernel install --user --name loan-approval --display-name "Loan Approval"
   ```
1. In VSCode Press `Ctrl+Shift+P` and select "Python: Select Interpreter", then choose the "Loan Approval" interpreter.
2. Once you open the Jupyter Notebook, it should automatically use the "Loan Approval" kernel. If not, please restart VSCode. And if not successful, then on the top right corner of the notebook, you can manually select the kernel by clicking on it and choosing "Loan Approval". You most likely will find it in the Jupyter kernel list.
   

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🅱️ Option 2: Run with Docker

Build and start the app using Docker:

```bash
docker build -t streamlit_app .
docker run --rm -p 8501:8501 streamlit_app
```

Or use the provided shell script:

```bash
bash start.sh
```

## 🛠️ Step-by-Step Guide

### 🏆 Kaggle Setup

To use Kaggle datasets or APIs, you need to set up your Kaggle credentials:

1. 🔗 Go to your Kaggle account settings: [https://www.kaggle.com/account](https://www.kaggle.com/account).
2. 🛡️ Scroll down to the **API** section and click **Create New API Token**.
3. 📥 This will download a file named `kaggle.json`.
4. 📂 Place `kaggle.json` in the folder:  
    - 🐧 **Linux/macOS:** `~/.kaggle/` or sometimes in `~/.config/kaggle/kaggle.json`
    - 🪟 **Windows:** `C:\Users\<YourUsername>\.kaggle\`
5. 🔒 Make sure the file permissions are secure (Linux/macOS):
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```
6. In order to be able to download the dataset you must join the competition and [accept the rules.](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

7. 📦 You can now use the Kaggle CLI to download datasets:
    ```bash
    kaggle competitions download -c house-prices-advanced-regression-techniques -p data/
    ```



### 1️⃣ Data Cleaning & Feature Engineering
- Open and run `data_cleaning_and_feature_engineering.ipynb` to preprocess and engineer features from the raw data.
- Follow up to train a Linear Regression Model on this difficult case.

### 2️⃣ Train Model (if needed)
- Use scripts or notebook to train and save models in `models/` (default model provided). Note: By running the `data_cleaning_and_feature_engineering.ipynb` notebook, it will train the model and save it automatically.

### 3️⃣ Launch the Streamlit App
- Run the following command or using docker:
    ```bash
    streamlit run main.py
    ```
- Use the sidebar to select between house price prediction and custom regression modules.
- Train your own model on your own data using the custom regression module.

   **Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Conda Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate kaggle-house-prices
   ```

3. **Download dataset** (requires Kaggle account)
   ```bash
   kaggle competitions download -c house-prices-advanced-regression-techniques -p data/
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t house-prices-app .
docker run --rm -p 8501:8501 house-prices-app

# Or use the provided script
bash start.sh
```

## 💻 Usage

### Web Application

1. **House Price Prediction**: Input property features to get price predictions
2. **Custom Model Training**: Upload your own dataset and train models interactively
3. **Model Comparison**: Compare performance between different algorithms

### Jupyter Notebook

Run `data_cleaning_and_feature_engineering.ipynb` to:
- Explore the dataset through comprehensive EDA
- Apply feature engineering techniques
- Train and evaluate machine learning models
- Generate submission files for Kaggle

## � Configuration

Key configuration options in `configs/config.py`:
- Model file paths
- Feature definitions and default values
- Application settings and parameters

## 📝 API Reference

### Model Schema (Pydantic)

The application uses Pydantic schemas for data validation:

```python
# Input features validation
class HouseFeatures(BaseModel):
    overall_qual: int
    gr_liv_area: float
    garage_cars: float
    # ... additional features

# Prediction output
class PricePrediction(BaseModel):
    predicted_price: float
    confidence_interval: Optional[Tuple[float, float]]
```

## � Model Performance Results

| Model | MAE | MSE | RMSE |
|-------|-----|-----|------|
| Linear Regression | 19,452.09 | 711,102,117.35 | 26,666.50 |
| **XGBoost** | **16,483.55** | **578,007,680.00** | **24,041.79** |

XGBoost demonstrates superior performance across all metrics, providing more accurate and robust predictions for house price estimation.


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) for providing the dataset and competition platform
- The Ames Housing dataset compiled by Dean De Cock
- Streamlit for the excellent web framework
- The open-source community for the amazing machine learning libraries

## 📚 Documentation

For detailed setup and learning instructions, see:
- [Student Instructions](Instructions.md) - Complete learning guide
- [Environment Setup](Docs/3.Setup_Environment.md) - Development environment
- [Kaggle Setup](Docs/4.Setup_Kaggle.md) - API and data access
- [GitHub Setup](Docs/1.Setup_Github.md) - Version control setup
